from __future__ import annotations

import argparse
from pathlib import Path

from src.document.chunker import estimate_tokens
from src.document.loader import load_document
from src.utils.file_utils import write_text
from src.utils.json_utils import dump_json
from src.utils.logging import configure_logging


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
        print(f"Overview: {run_dir / 'overview.puml'}")
        print(f"Full diagram: {run_dir / 'full.puml'}")
        print(f"Report: {run_dir / 'report.md'}")
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
    print(f"PlantUML: {out / f'{sample_id}.puml'}")


if __name__ == "__main__":
    main()
