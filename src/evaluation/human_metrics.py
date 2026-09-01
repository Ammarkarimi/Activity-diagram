from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from collections import defaultdict


DIMENSIONS = [
    "semantic_correctness",
    "completeness",
    "structural_correctness",
    "readability",
    "abstraction",
    "overall_quality",
]


def summarize_human_ratings(csv_path: str | Path):
    groups = defaultdict(list)
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["sample_id"]].append(row)
    return {
        sample: {
            dim: mean(float(r[dim]) for r in rows)
            for dim in DIMENSIONS
        }
        for sample, rows in groups.items()
    }
