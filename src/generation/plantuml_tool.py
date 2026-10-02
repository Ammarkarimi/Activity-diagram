"""Locate and run PlantUML: ``plantuml.jar`` with Java, or a ``plantuml`` executable.

The jar is looked up in ``$PLANTUML_JAR``, then ``tools/plantuml.jar`` in the
project (``python -m scripts.install_plantuml`` downloads it there). Java is
taken from ``$JAVA_HOME`` or ``PATH``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JAR = PROJECT_ROOT / "tools" / "plantuml.jar"

# PlantUML crops PNG output larger than this many pixels in either direction.
# The pipeline raises the limit for its own renders, but editors and the
# public PlantUML server keep the default, so diagrams above it are reported.
DEFAULT_LIMIT_SIZE = 4096
RENDER_LIMIT_SIZE = 32768


def _java() -> str | None:
    home = os.getenv("JAVA_HOME")
    if home:
        for name in ("java.exe", "java"):
            candidate = Path(home) / "bin" / name
            if candidate.exists():
                return str(candidate)
    return shutil.which("java")


def plantuml_command() -> list[str] | None:
    """The command prefix that runs PlantUML, or None when it is not installed."""
    jar = os.getenv("PLANTUML_JAR") or str(DEFAULT_JAR)
    if Path(jar).is_file():
        java = _java()
        if java:
            return [java, "-Djava.awt.headless=true", "-jar", jar]
    exe = shutil.which("plantuml")
    return [exe] if exe else None


def run_plantuml(
    args: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str] | None:
    """Run PlantUML with ``args``; None when PlantUML is not available."""
    command = plantuml_command()
    if command is None:
        return None
    env = dict(os.environ, PLANTUML_LIMIT_SIZE=str(RENDER_LIMIT_SIZE))
    return subprocess.run(
        [*command, "-charset", "UTF-8", *args],
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=False,
    )
