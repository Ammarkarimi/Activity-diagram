"""Download plantuml.jar into tools/ so the pipeline can check and render diagrams.

usage: python -m scripts.install_plantuml [--version 1.2024.7]

Java must be installed ($JAVA_HOME or `java` on PATH).
"""
from __future__ import annotations

import argparse
import urllib.request

from src.generation.plantuml_tool import DEFAULT_JAR, plantuml_command, run_plantuml

URL = "https://github.com/plantuml/plantuml/releases/download/v{version}/plantuml-{version}.jar"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="1.2024.7")
    args = parser.parse_args()

    DEFAULT_JAR.parent.mkdir(parents=True, exist_ok=True)
    url = URL.format(version=args.version)
    print(f"Downloading {url}")
    partial = DEFAULT_JAR.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(DEFAULT_JAR)
    print(f"Saved {DEFAULT_JAR}")

    if plantuml_command() is None:
        print("Java not found: set JAVA_HOME or put java on PATH.")
        return
    proc = run_plantuml(["-version"])
    print(proc.stdout.splitlines()[0] if proc.stdout else proc.stderr)


if __name__ == "__main__":
    main()
