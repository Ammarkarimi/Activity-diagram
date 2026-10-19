"""Download plantuml.jar into tools/ so the pipeline can check and render diagrams.

usage: python -m scripts.install_plantuml [--version 1.2024.7]

Java must be installed ($JAVA_HOME or `java` on PATH).
"""
from __future__ import annotations

import argparse

from src.generation.plantuml_tool import (
    JAR_URL,
    PLANTUML_VERSION,
    download_jar,
    plantuml_command,
    run_plantuml,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=PLANTUML_VERSION)
    args = parser.parse_args()

    print(f"Downloading {JAR_URL.format(version=args.version)}")
    print(f"Saved {download_jar(args.version)}")

    if plantuml_command() is None:
        print("Java not found: set JAVA_HOME or put java on PATH.")
        return
    proc = run_plantuml(["-version"])
    print(proc.stdout.splitlines()[0] if proc.stdout else proc.stderr)


if __name__ == "__main__":
    main()
