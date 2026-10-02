from __future__ import annotations

import argparse
from pathlib import Path

from src.document.chunker import estimate_tokens
from src.document.loader import load_document
from src.utils.file_utils import write_text
from src.utils.json_utils import dump_json
from src.utils.logging import configure_logging


def _print_usage(metrics: dict) -> None:
    total = metrics.get("token_usage", {}).get("total")
    if not total:
        return
    cost = f"${total['cost_usd']:.4f}" if total.get("cost_usd") is not None else "n/a (model not in configs/pricing.json)"
    print(f"Tokens: {total['total_tokens']:,} ({total['input_tokens']:,} in, "
          f"{total['output_tokens']:,} out) in {total['calls']} calls; cost: {cost}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate UML activity diagrams from natural-language requirements."
    )
    parser.add_argument("--input", required=True, help=".txt, .md, .pdf or .docx specification")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--mode",
        choices=["auto", "single", "hierarchical"],
        default="auto",
        help="single: one diagram from the whole text; hierarchical: chunked "
        "extraction + per-module diagrams (for long documents); auto picks "
        "hierarchical above --long-doc-tokens.",
    )
    parser.add_argument("--long-doc-tokens", type=int, default=3000)
    parser.add_argument("--chunk-tokens", type=int, default=2500)
    parser.add_argument("--max-requirements-per-module", type=int, default=25)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    configure_logging()

    input_path = Path(args.input)
    sample_id = args.sample_id or input_path.stem
    text = load_document(input_path)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    if mode == "auto":
        mode = "hierarchical" if estimate_tokens(text) > args.long_doc_tokens else "single"

    if mode == "hierarchical":
        from src.pipeline.hierarchical import HierarchicalPipeline

        state = HierarchicalPipeline(
            model=args.model,
            chunk_tokens=args.chunk_tokens,
            max_requirements_per_module=args.max_requirements_per_module,
            max_workers=args.max_workers,
        ).run(
            sample_id=sample_id,
            requirement_text=text,
            max_iterations=args.max_iterations,
            output_dir=out,
            document_title=input_path.stem,
        )
        run_dir = Path(state.metrics["run_output_dir"])
        print(f"Completed: {sample_id} (hierarchical)")
        print(f"Requirements: {len(state.requirements)} in {len(state.modules)} modules")
        print(f"Requirement coverage: {100 * state.metrics['requirement_coverage']:.1f}%")
        errors = state.metrics.get("plantuml_errors", {})
        checks = state.metrics.get("plantuml_checks", {})
        for name, label in (("overview", "Overview"), ("full", "Full diagram")):
            if name in errors:
                print(f"{label}: FAILED ({errors[name]})")
                continue
            print(f"{label}: {run_dir / f'{name}.puml'}")
            check = checks.get(name, {})
            if check.get("note"):
                print(f"  {check['note']}")
            if check.get("oversized"):
                print(f"  {check['width']} x {check['height']} px: open the .svg, PNG previews crop at 4096 px.")
        parts = state.metrics.get("parts", [])
        if parts:
            print(f"Parts: {sum(1 for p in parts if p.get('stem'))} linked diagrams in {run_dir / 'parts'}")
        print(f"Swimlanes: {state.metrics.get('full_diagram_lanes', 0)} "
              f"({', '.join(state.metrics.get('actors', [])) or 'not harmonised'})")
        module_errors = sorted(set(errors) - {"overview", "full"})
        if module_errors:
            print(f"Diagrams with PlantUML errors: {', '.join(module_errors)}")
        if checks and not any(c["verified"] for c in checks.values()):
            print("PlantUML not found: syntax was checked with built-in rules only "
                  "(run: python -m scripts.install_plantuml).")
        _print_usage(state.metrics)
        print(f"Report: {run_dir / 'report.md'}")
        print(f"Viewer: {run_dir / 'viewer.html'} (open in a browser)")
        return

    from src.pipeline.orchestrator import MultiAgentPipeline

    state = MultiAgentPipeline(model=args.model).run(
        sample_id=sample_id,
        requirement_text=text,
        max_iterations=args.max_iterations,
        output_dir=out,
    )
    dump_json(state.model_dump(mode="json"), out / f"{sample_id}_state.json")
    write_text(state.final_plantuml, out / f"{sample_id}.puml")

    print(f"Completed: {sample_id}")
    print(f"Iterations: {state.iteration}")
    print(f"Defects remaining: {len(state.defects)}")
    _print_usage(state.metrics)
    for entry in state.metrics.get("candidate_scores", []):
        cost = entry.get("iteration_cost_usd")
        print(f"  iteration {entry['iteration']}: {entry.get('iteration_tokens', 0):,} tokens"
              + (f", ${cost:.4f}" if cost is not None else ""))
    print(f"PlantUML: {out / f'{sample_id}.puml'}")


if __name__ == "__main__":
    main()
