from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class PlantUMLSyntaxValidator:

    def __init__(
        self,
        plantuml_command: str = "plantuml",
    ):
        self.plantuml_command = plantuml_command

    def validate(
        self,
        plantuml_text: str,
    ) -> tuple[bool, str]:

        with tempfile.TemporaryDirectory() as tmp:

            source = Path(tmp) / "diagram.puml"

            source.write_text(
                plantuml_text,
                encoding="utf-8",
            )

            try:

                process = subprocess.run(
                    [
                        self.plantuml_command,
                        "-checkmetadata",
                        str(source),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            except FileNotFoundError:

                return (
                    False,
                    "PlantUML executable not found."
                )

            except subprocess.TimeoutExpired:

                return (
                    False,
                    "PlantUML validation timed out."
                )

            if process.returncode == 0:

                return (
                    True,
                    process.stdout
                )

            return (
                False,
                process.stderr or process.stdout
            )