from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = Path(args.input).read_text(encoding="utf-8")
    out = {
        "source_file": args.input,
        "text": raw,
        "notes": [
            "Replace this script with the exact PURE/PAGED ingestion logic used in the study.",
            "Keep the original files under data/raw/ untouched.",
        ],
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    main()
