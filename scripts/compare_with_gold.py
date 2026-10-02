"""Compare generated activity diagrams with the gold-standard solutions.

Both diagrams are parsed with the same PlantUML parser and scored with the
same checks, so neither side gets an advantage from its source format.

Metrics
-------
- Structural validity: the deterministic StructuralValidator score.
- Action recall / precision / F1: one-to-one label matching between gold
  and generated actions (label similarity >= threshold).
- Ordering agreement: for each pair of consecutive gold actions (b directly
  follows a, ignoring control nodes) whose actions were both matched, does
  the generated diagram also reach match(b) from match(a)?
- Vacuous decisions: decisions where every branch is empty (no behaviour
  in any branch) -- a modelling defect.
- Size and complexity: actions, decisions, forks, lanes, cyclomatic
  complexity (E - N + 2).

Usage:
    python -m scripts.compare_with_gold
    python -m scripts.compare_with_gold --generated outputs --gold "Gold Standard Solutions"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import networkx as nx

from src.evaluation.semantic_metrics import label_similarity
from src.models.domain import ActivityDiagram, NodeType
from src.parsing.activity_parser import ActivityPlantUMLParser
from src.validation.structural.structural_validator import StructuralValidator


def sample_key(name: str) -> str:
    name = re.sub(r"^\d{4}\s*-\s*", "", Path(name).stem)
    name = re.sub(r"_Run\d+$", "", name, flags=re.IGNORECASE)
    # Hierarchical runs: <sample>_full.puml is the complete diagram.
    name = re.sub(r"_full$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def actions(diagram: ActivityDiagram):
    return [n for n in diagram.nodes if n.type == NodeType.ACTION]


def match_actions(gold: ActivityDiagram, generated: ActivityDiagram, threshold: float):
    pairs = []
    for g in actions(gold):
        for c in actions(generated):
            score = label_similarity(g.label, c.label)
            if score >= threshold:
                pairs.append((score, g.id, c.id))
    pairs.sort(reverse=True)
    matched_gold, matched_gen, mapping = set(), set(), {}
    for score, g, c in pairs:
        if g in matched_gold or c in matched_gen:
            continue
        matched_gold.add(g)
        matched_gen.add(c)
        mapping[g] = (c, score)
    return mapping


def graph(diagram: ActivityDiagram) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in diagram.nodes:
        g.add_node(n.id, type=n.type)
    for e in diagram.edges:
        g.add_edge(e.source, e.target)
    return g


def action_successor_pairs(diagram: ActivityDiagram) -> set[tuple[str, str]]:
    """(a, b) where action b follows action a through control nodes only."""
    g = graph(diagram)
    action_ids = {n.id for n in actions(diagram)}
    pairs = set()
    for a in action_ids:
        stack, seen = list(g.successors(a)), set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in action_ids:
                pairs.add((a, node))
            else:
                stack.extend(g.successors(node))
    return pairs


def vacuous_decisions(diagram: ActivityDiagram) -> int:
    g = graph(diagram)
    count = 0
    for n in diagram.nodes:
        if n.type != NodeType.DECISION:
            continue
        targets = set(g.successors(n.id))
        if len(targets) == 1 and g.out_degree(n.id) >= 1:
            target = next(iter(targets))
            if g.nodes[target]["type"] in {NodeType.MERGE, NodeType.FINAL}:
                count += 1
    return count


def profile(diagram: ActivityDiagram) -> dict:
    validation = StructuralValidator().validate(diagram)
    count = lambda t: sum(1 for n in diagram.nodes if n.type == t)
    return {
        "structural_score": round(validation.score, 3),
        "structural_defects": len(validation.defects),
        "actions": count(NodeType.ACTION),
        "decisions": count(NodeType.DECISION),
        "forks": count(NodeType.FORK),
        "finals": count(NodeType.FINAL),
        "lanes": len({n.lane for n in diagram.nodes if n.lane}),
        "cyclomatic": len(diagram.edges) - len(diagram.nodes) + 2,
        "vacuous_decisions": vacuous_decisions(diagram),
    }


def compare(gold: ActivityDiagram, generated: ActivityDiagram, threshold: float, mapping=None) -> dict:
    """``mapping`` (gold ID -> (generated ID, score)) defaults to label matching."""
    if mapping is None:
        mapping = match_actions(gold, generated, threshold)
    n_gold, n_gen = len(actions(gold)), len(actions(generated))
    recall = len(mapping) / n_gold if n_gold else 0.0
    precision = len(mapping) / n_gen if n_gen else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    gen_graph = graph(generated)
    checked = agreed = 0
    for a, b in action_successor_pairs(gold):
        if a in mapping and b in mapping:
            checked += 1
            if nx.has_path(gen_graph, mapping[a][0], mapping[b][0]):
                agreed += 1

    gold_labels = {n.id: n.label for n in gold.nodes}
    gen_labels = {n.id: n.label for n in generated.nodes}
    return {
        "action_recall": round(recall, 3),
        "action_precision": round(precision, 3),
        "action_f1": round(f1, 3),
        "ordering_agreement": round(agreed / checked, 3) if checked else None,
        "ordering_pairs_checked": checked,
        "missing_gold_actions": [gold_labels[n.id] for n in actions(gold) if n.id not in mapping],
        "extra_generated_actions": [
            gen_labels[n.id] for n in actions(generated) if n.id not in {c for c, _ in mapping.values()}
        ],
        "matches": [
            {"gold": gold_labels[g], "generated": gen_labels[c], "similarity": round(s, 2)}
            for g, (c, s) in mapping.items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="Gold Standard Solutions")
    parser.add_argument("--generated", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="results/gold_comparison")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="also pair steps by meaning with an LLM judge (majority of 5 judgements)",
    )
    parser.add_argument("--judge-model", default=None, help="judge model (default: OPENAI_MODEL)")
    args = parser.parse_args()
    llm = None
    if args.semantic:
        from src.llm.openai_client import OpenAIClient

        llm = OpenAIClient(model=args.judge_model)

    puml = ActivityPlantUMLParser()
    gold = {
        sample_key(p.name): (p, puml.parse(p.read_text(encoding="utf-8")))
        for p in sorted(Path(args.gold).glob("*.txt"))
    }

    results = []
    for path in sorted(Path(args.generated).rglob("*.puml")):
        key = sample_key(path.name)
        text = path.read_text(encoding="utf-8")
        if key not in gold or not text.strip():
            continue
        gold_path, gold_diagram = gold[key]
        generated = puml.parse(text)
        results.append(
            {
                "sample": key,
                "generated_file": str(path),
                "gold_file": str(gold_path),
                "gold": profile(gold_diagram),
                "generated": profile(generated),
                "similarity": compare(gold_diagram, generated, args.threshold),
            }
        )
        if llm is not None:
            from src.evaluation.semantic_alignment import semantic_mapping

            mapping = semantic_mapping(gold_diagram, generated, llm)
            results[-1]["semantic"] = compare(gold_diagram, generated, args.threshold, mapping)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    header = (
        "| Sample | Generated file | Action F1 | Recall | Precision | Ordering | "
        "Struct. gold/gen | Actions gold/gen | Decisions gold/gen | Lanes gold/gen | Vacuous dec. gold/gen |"
    )
    lines = ["# Generated vs gold-standard activity diagrams", "", header, "|" + "---|" * 11]
    for r in results:
        g, c, s = r["gold"], r["generated"], r["similarity"]
        lines.append(
            f"| {r['sample']} | {r['generated_file']} | {s['action_f1']} | {s['action_recall']} | "
            f"{s['action_precision']} | {s['ordering_agreement']} | {g['structural_score']}/{c['structural_score']} | "
            f"{g['actions']}/{c['actions']} | {g['decisions']}/{c['decisions']} | {g['lanes']}/{c['lanes']} | "
            f"{g['vacuous_decisions']}/{c['vacuous_decisions']} |"
        )
    # Several runs per sample: mean and standard deviation.
    by_sample: dict[str, list[dict]] = {}
    for r in results:
        by_sample.setdefault(r["sample"], []).append(r)

    def spread(values) -> str:
        values = [v for v in values if v is not None]
        if not values:
            return "-"
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
        return f"{mean:.2f} ± {std:.2f}"

    lines += [
        "",
        "## Summary per sample (mean ± std over runs)",
        "",
        "| Sample | Runs | Action F1 | Semantic F1 | Semantic ordering | Lanes (gold) | Decisions (gold) |",
        "|---|---|---|---|---|---|---|",
    ]
    for sample, runs in by_sample.items():
        lines.append(
            f"| {sample} | {len(runs)} | {spread(r['similarity']['action_f1'] for r in runs)} | "
            f"{spread(r['semantic']['action_f1'] for r in runs if 'semantic' in r)} | "
            f"{spread(r['semantic']['ordering_agreement'] for r in runs if 'semantic' in r)} | "
            f"{spread(r['generated']['lanes'] for r in runs)} ({runs[0]['gold']['lanes']}) | "
            f"{spread(r['generated']['decisions'] for r in runs)} ({runs[0]['gold']['decisions']}) |"
        )

    semantic = [r for r in results if "semantic" in r]
    if semantic:
        lines += [
            "",
            "## Steps paired by meaning (LLM judge)",
            "",
            "| Sample | Generated file | Action F1 | Recall | Precision | Ordering |",
            "|---|---|---|---|---|---|",
        ]
        for r in semantic:
            s = r["semantic"]
            lines.append(
                f"| {r['sample']} | {r['generated_file']} | {s['action_f1']} | {s['action_recall']} | "
                f"{s['action_precision']} | {s['ordering_agreement']} |"
            )
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nDetails: {out / 'comparison.json'}")


if __name__ == "__main__":
    main()
