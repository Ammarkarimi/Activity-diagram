from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baselines.cot_baseline import run as cot_run
from src.baselines.direct_llm import run as direct_run
from src.baselines.ladex_baseline import run as ladex_run
from src.baselines.self_refine import run as self_refine_run
from src.evaluation.evaluator import evaluate
from src.pipeline.orchestrator import MultiAgentPipeline
from src.models.domain import Requirement
from src.utils.json_utils import dump_json


def load_dataset(path: str):
    p = Path(path)
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_gold(gold_path: str):
    from src.models.domain import ActivityDiagram
    obj = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    return ActivityDiagram.model_validate(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="JSONL with sample_id, requirement_text, requirements, gold_diagram_path")
    parser.add_argument("--experiment", choices=["direct_llm", "cot", "self_refine", "ladex", "multi_agent"], required=True)
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in load_dataset(args.dataset):
        text = row["requirement_text"]
        sample_id = row["sample_id"]

        if args.experiment == "direct_llm":
            diagram = direct_run(text, model=args.model)
        elif args.experiment == "cot":
            diagram = cot_run(text, model=args.model)
        elif args.experiment == "self_refine":
            diagram = self_refine_run(text, model=args.model)
        elif args.experiment == "ladex":
            diagram = ladex_run(text, model=args.model)
        else:
            state = MultiAgentPipeline(model=args.model).run(
                sample_id=sample_id,
                requirement_text=text,
                max_iterations=3,
                output_dir=out_dir,
            )
            diagram = state.diagram

        gold = load_gold(row["gold_diagram_path"])
        reqs = [Requirement.model_validate(x) for x in row.get("requirements", [])]

        metrics = evaluate(diagram, gold, reqs)
        dump_json(
            {
                "sample_id": sample_id,
                "experiment": args.experiment,
                "diagram": diagram.model_dump(mode="json"),
                "metrics": metrics,
            },
            out_dir / f"{sample_id}.json",
        )

    print(f"Finished {args.experiment}: {out_dir}")


if __name__ == "__main__":
    main()
