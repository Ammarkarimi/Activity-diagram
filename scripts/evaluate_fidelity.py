"""Score diagrams against the SPECIFICATION, the same way for gold and generated ones.

Gold-referenced metrics (compare_with_gold) can only measure how close a
diagram is to one expert's drawing; the gold itself scores 1.0 by
definition. This script asks a different question of every diagram,
including the gold standard:

- behaviour coverage: share of the behaviours the specification states that
  the diagram shows;
- unsupported steps: share of the diagram's actions and decisions that the
  specification does not support;
- fidelity: harmonic mean of coverage and (1 - unsupported).

The behaviour list is built once per specification with an evaluation-only
prompt (not the pipeline's requirement agent) and cached, so every diagram
is judged against the same list. Every judgement is a majority vote.

Usage:
    python -m scripts.evaluate_fidelity --generated results/raw/v3 --output results/raw/fidelity_v3
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from scripts.compare_with_gold import sample_key
from src.llm.openai_client import OpenAIClient

SPECS = {"agentmom": "agentmom.txt", "central_trading": "central_trading.txt", "peering": "peering.txt"}


class Behaviours(BaseModel):
    behaviours: list[str] = Field(default_factory=list)


class Indices(BaseModel):
    indices: list[int] = Field(default_factory=list)


_BEHAVIOURS = """
List every behaviour the specification states, one short sentence each:
steps that an actor or the system performs, conditions and decisions with
their outcomes, exceptions and how they end, loops, and parallel work.
Use the specification's own terms. Skip goals, benefits, quality attributes,
glossary entries, open issues and questions, and illustrative examples.
Do not add anything the text does not state.
"""

_COVERED = """
You check which behaviours of a specification an activity diagram shows.
A behaviour is shown when the diagram contains a step, decision or branch
with the same meaning, even if worded differently or combined with other
behaviour in one step. It is not shown if the diagram only mentions related
words. Return the numbers of the behaviours that are shown.
"""

_UNSUPPORTED = """
You check an activity diagram against its specification. Return the numbers
of the diagram STEPS that the specification does not support.

Supported (do NOT return):
- a step stated anywhere in the specification, including introduction,
  context, user and actor sections, in any wording, summarised, or combined
  with other stated steps;
- a decision that tests a condition, choice, exception or outcome the
  specification mentions ("Message encrypted?", "Instruction type?");
- a step that is a direct part of stated behaviour (receiving a message the
  text says is sent, reporting a result the text says is produced).

Unsupported (return): behaviour the text never states or implies, such as
invented steps, technologies, protocols, parameters or outcomes, and
non-steps drawn as steps (goals, benefits, quality attributes).
"""


def diagram_steps(plantuml: str) -> list[str]:
    """Actions and decisions of a PlantUML activity diagram, in order."""
    steps, action = [], None
    for raw in plantuml.splitlines():
        line = raw.strip()
        if action is not None:
            action.append(line)
            if line.endswith((";", "|", "<", ">", "]", "}", "/")):
                steps.append(" ".join(action)[1:-1])
                action = None
            continue
        if line.startswith(":") and not line.startswith("::"):
            if line.endswith((";", "|", "<", ">", "]", "}", "/")) and len(line) > 2:
                steps.append(line[1:-1])
            else:
                action = [line]
        # "elseif" repeats its decision's question: count the decision once.
        elif line.startswith(("if (", "while (", "repeat while (")):
            steps.append("Decision: " + line[line.index("(") + 1:line.index(")")])
    return [" ".join(s.replace("\\n", " ").split()) for s in steps if s.strip()]


def vote(llm, system: str, user: str, limit: int, votes: int) -> set[int]:
    counts: Counter[int] = Counter()
    for _ in range(votes):
        result = llm.complete(system=system, user=user, response_model=Indices)
        counts.update({i for i in result.indices if 1 <= i <= limit})
    return {i for i, c in counts.items() if c * 2 > votes}


def numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def behaviours_for(sample: str, spec: str, llm, cache: Path) -> list[str]:
    path = cache / f"{sample}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = llm.complete(system=_BEHAVIOURS, user=f"SPECIFICATION\n{spec}", response_model=Behaviours)
    behaviours = [b.strip() for b in result.behaviours if b.strip()]
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(behaviours, indent=2, ensure_ascii=False), encoding="utf-8")
    return behaviours


def score(plantuml: str, spec: str, behaviours: list[str], llm, votes: int) -> dict:
    steps = diagram_steps(plantuml)
    covered = vote(
        llm, _COVERED,
        f"BEHAVIOURS\n{numbered(behaviours)}\n\nDIAGRAM (PlantUML)\n{plantuml}",
        len(behaviours), votes,
    )
    unsupported = vote(
        llm, _UNSUPPORTED,
        f"SPECIFICATION\n{spec}\n\nDIAGRAM STEPS\n{numbered(steps)}",
        len(steps), votes,
    )
    coverage = len(covered) / len(behaviours) if behaviours else 0.0
    unsupported_rate = len(unsupported) / len(steps) if steps else 0.0
    support = 1 - unsupported_rate
    fidelity = 2 * coverage * support / (coverage + support) if coverage + support else 0.0
    return {
        "steps": len(steps),
        "behaviour_coverage": round(coverage, 3),
        "unsupported_rate": round(unsupported_rate, 3),
        "fidelity": round(fidelity, 3),
        "missing_behaviours": [b for i, b in enumerate(behaviours, start=1) if i not in covered],
        "unsupported_steps": [s for i, s in enumerate(steps, start=1) if i in unsupported],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="Gold Standard Solutions")
    parser.add_argument("--generated", required=True)
    parser.add_argument("--specs", default="data/pure")
    parser.add_argument("--output", default="results/fidelity")
    parser.add_argument("--cache", default="results/reference_behaviours")
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--model", default=None, help="judge model (default: OPENAI_MODEL)")
    args = parser.parse_args()

    llm = OpenAIClient(model=args.model)
    files = [(p, "gold") for p in sorted(Path(args.gold).glob("*.txt"))]
    files += [(p, "generated") for p in sorted(Path(args.generated).rglob("*.puml"))]
    results = []
    for path, kind in files:
        key = sample_key(path.name)
        if key not in SPECS or path.parent.name in ("modules", "parts"):
            continue
        spec = (Path(args.specs) / SPECS[key]).read_text(encoding="utf-8")
        behaviours = behaviours_for(key, spec, llm, Path(args.cache))
        result = score(path.read_text(encoding="utf-8"), spec, behaviours, llm, args.votes)
        results.append({"sample": key, "kind": kind, "file": str(path), **result})
        print(f"{key:16} {kind:9} coverage={result['behaviour_coverage']:.2f} "
              f"unsupported={result['unsupported_rate']:.2f} fidelity={result['fidelity']:.2f}  {path}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fidelity.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Fidelity to the specification (gold and generated judged the same way)",
        "",
        "| Sample | Diagrams | Runs | Behaviour coverage | Unsupported steps | Fidelity |",
        "|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        groups.setdefault((r["sample"], r["kind"]), []).append(r)

    def spread(values: list[float]) -> str:
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
        return f"{mean:.2f} ± {std:.2f}"

    for (sample, kind), runs in sorted(groups.items()):
        lines.append(
            f"| {sample} | {kind} | {len(runs)} | {spread([r['behaviour_coverage'] for r in runs])} | "
            f"{spread([r['unsupported_rate'] for r in runs])} | {spread([r['fidelity'] for r in runs])} |"
        )
    (out / "fidelity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
