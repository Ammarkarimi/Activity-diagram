from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.orchestrator import MultiAgentPipeline
from src.utils.file_utils import read_text, write_text
from src.utils.json_utils import dump_json
from src.utils.logging import configure_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    configure_logging()

    input_path = Path(args.input)
    sample_id = args.sample_id or input_path.stem
    text = read_text(input_path)

    state = MultiAgentPipeline(model=args.model).run(
        sample_id=sample_id,
        requirement_text=text,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(state.model_dump(mode="json"), out / f"{sample_id}_state.json")
    write_text(state.final_plantuml, out / f"{sample_id}.puml")

    print(f"Completed: {sample_id}")
    print(f"Iterations: {state.iteration}")
    print(f"Defects remaining: {len(state.defects)}")
    print(f"PlantUML: {out / f'{sample_id}.puml'}")


if __name__ == "__main__":
    main()
