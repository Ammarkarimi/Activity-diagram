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

    def _regex_validate(self, plantuml_text: str) -> tuple[bool, str]:
        import re
        text = plantuml_text.strip()
        if not text.startswith("@startuml"):
            return False, "Missing @startuml at the beginning"
        if not text.endswith("@enduml"):
            return False, "Missing @enduml at the end"
            
        if len(re.findall(r'\bif\b', plantuml_text)) != len(re.findall(r'\bendif\b', plantuml_text)):
            return False, "Unbalanced if/endif pairs"
            
        forks = len(re.findall(r'^[ \t]*fork[ \t]*$', plantuml_text, re.MULTILINE))
        end_forks = len(re.findall(r'^[ \t]*end fork[ \t]*$', plantuml_text, re.MULTILINE))
        if forks != end_forks:
            return False, "Unbalanced fork/end fork pairs"
            
        repeats = len(re.findall(r'^[ \t]*repeat[ \t]*$', plantuml_text, re.MULTILINE))
        repeat_whiles = len(re.findall(r'^[ \t]*repeat while\b', plantuml_text, re.MULTILINE))
        if repeats != repeat_whiles:
            return False, "Unbalanced repeat/repeat while pairs"
            
        for line in plantuml_text.splitlines():
            line = line.strip()
            if line.startswith(':') and not line.endswith(';'):
                return False, "Action lines starting with : must end with ;"
                
        if re.search(r'if\s*\(\s*\)', plantuml_text):
            return False, "Empty if() condition found"
            
        return True, "Regex validation passed"

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
                        "-syntax",
                        str(source),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            except FileNotFoundError:

                return self._regex_validate(plantuml_text)

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