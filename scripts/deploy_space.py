"""Upload the tool to a Hugging Face Space (Docker SDK).

usage: python -m scripts.deploy_space --space <user>/<space-name>

Needs `pip install huggingface_hub` and a one-time `hf auth login` with a
Hugging Face token that has write access. Only files
committed to git are uploaded (in their current state), together with the
Space README that tells Hugging Face to build the Dockerfile. Running the
script again updates the Space.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
GITHUB_URL = "https://github.com/Ammarkarimi/Activity-diagram"

SPACE_README = f"""---
title: UML Activity Diagram Generator
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Multi-agent UML activity diagrams from requirements
---

Generate–Verify–Repair: multi-agent generation of UML activity diagrams from
natural-language requirements.

Enter your OpenAI API key in the sidebar to generate diagrams. The runs
reported in the paper open without a key: choose *Saved run* as the
requirements source.

Source code, data and setup guide: {GITHUB_URL}
"""


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [f for f in out.split("\0") if f and (ROOT / f).is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", required=True, help="<user>/<space-name>")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        files = tracked_files()
        for rel in files:
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, target)
        (stage / "README.md").write_text(SPACE_README, encoding="utf-8")
        print(f"Uploading {len(files)} files to {args.space} ...")
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=stage,
            commit_message="Update app",
            delete_patterns=["*"],  # drop files that were removed from the repository
        )
    print(f"Done. The Space builds now: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
