from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--output", default="results/metrics")
    args = parser.parse_args()

    root = Path(args.results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in root.rglob("*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = obj.get("metrics")
        if metrics:
            row = {"experiment": obj.get("experiment"), "sample_id": obj.get("sample_id")}
            row.update(metrics)
            rows.append(row)

    if not rows:
        print("No metric JSON files found.")
        return

    columns = sorted({k for row in rows for k in row})
    with (output / "all_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    numeric = [c for c in columns if c not in {"experiment", "sample_id"}]
    summaries = []
    experiments = sorted({r["experiment"] for r in rows})
    for exp in experiments:
        exp_rows = [r for r in rows if r["experiment"] == exp]
        summary = {"experiment": exp, "n": len(exp_rows)}
        for c in numeric:
            vals = [float(r[c]) for r in exp_rows if isinstance(r.get(c), (int, float))]
            if vals:
                summary[c] = mean(vals)
        summaries.append(summary)

    scols = sorted({k for row in summaries for k in row})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scols)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote: {output / 'all_samples.csv'}")
    print(f"Wrote: {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
