from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.orchestrator import MultiAgentPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/batch")
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    pipeline = MultiAgentPipeline()
    for path in sorted(Path(args.input_dir).glob("*.txt")):
        pipeline.run(
            sample_id=path.stem,
            requirement_text=path.read_text(encoding="utf-8"),
            max_iterations=args.max_iterations,
            output_dir=args.output_dir,
        )
        print(f"Generated {path.stem}")


if __name__ == "__main__":
    main()
