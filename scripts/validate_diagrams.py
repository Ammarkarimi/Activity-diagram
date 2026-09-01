from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.domain import ActivityDiagram, Requirement
from src.validation.validator import HybridValidator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagram", required=True)
    parser.add_argument("--requirements", required=True)
    args = parser.parse_args()

    diagram = ActivityDiagram.model_validate(json.loads(Path(args.diagram).read_text(encoding="utf-8")))
    requirements = [
        Requirement.model_validate(x)
        for x in json.loads(Path(args.requirements).read_text(encoding="utf-8"))
    ]

    result = HybridValidator().validate(diagram, requirements)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
