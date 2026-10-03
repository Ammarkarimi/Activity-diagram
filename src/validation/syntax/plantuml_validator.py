from __future__ import annotations

import html
import re
import subprocess
from dataclasses import dataclass

from src.generation.plantuml_generator import PlantUMLGenerator
from src.generation.plantuml_tool import DEFAULT_LIMIT_SIZE, run_plantuml

_VIEWBOX = re.compile(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"')
# The first line of the error image PlantUML draws when layout crashes.
_RENDER_ERROR = re.compile(r"An error has occur+ed\s*:?\s*([^<]*)")


@dataclass
class PlantUMLCheck:
    valid: bool
    message: str = ""
    # 1-based line of the first syntax error.
    line: int | None = None
    # True when PlantUML itself parsed the text; False when it is not
    # installed and only the built-in checks ran.
    verified: bool = False
    # Size of the rendered diagram in pixels (known only when verified).
    width: int | None = None
    height: int | None = None

    @property
    def oversized(self) -> bool:
        """Larger than PlantUML's default PNG limit: most viewers crop it."""
        return max(self.width or 0, self.height or 0) > DEFAULT_LIMIT_SIZE

    @property
    def size(self) -> str:
        return f"{self.width} x {self.height} px" if self.width else "unknown"


class PlantUMLSyntaxValidator:
    """Checks PlantUML text with PlantUML itself, or built-in checks when it is missing."""

    def check(self, plantuml_text: str) -> PlantUMLCheck:
        try:
            process = run_plantuml(["-tsvg", "-pipe"], input_text=plantuml_text)
        except subprocess.TimeoutExpired:
            return PlantUMLCheck(valid=False, message="PlantUML validation timed out.")
        except OSError as exc:
            process = None
            fallback = f" (PlantUML could not be started: {exc})"
        else:
            fallback = " (PlantUML not installed; built-in checks only)"

        if process is None:
            valid, message = self._regex_validate(plantuml_text)
            return PlantUMLCheck(valid=valid, message=message + fallback)

        if process.returncode != 0:
            # stderr is "ERROR", the 0-based line, then the message.
            parts = [p.strip() for p in process.stderr.splitlines() if p.strip()]
            line = None
            if len(parts) >= 2 and parts[0] == "ERROR" and parts[1].isdigit():
                line = int(parts[1]) + 1
                message = parts[2] if len(parts) > 2 else "Syntax error"
            else:
                message = process.stderr.strip() or f"PlantUML exited with code {process.returncode}"
            if line is not None:
                source = plantuml_text.splitlines()
                if line <= len(source):
                    message = f"line {line}: {message}: {source[line - 1].strip()}"
            return PlantUMLCheck(valid=False, message=message, line=line, verified=True)

        # A layout crash (e.g. a NullPointerException while drawing) still
        # exits with 0: PlantUML returns an image of the error instead.
        crash = _RENDER_ERROR.search(process.stdout)
        if crash:
            return PlantUMLCheck(
                valid=False,
                message=f"PlantUML failed to draw the diagram: {html.unescape(crash.group(1)).strip()}",
                verified=True,
            )

        match = _VIEWBOX.search(process.stdout[:2000])
        width, height = (round(float(v)) for v in match.groups()) if match else (None, None)
        return PlantUMLCheck(
            valid=True,
            message="PlantUML parsed the diagram",
            verified=True,
            width=width,
            height=height,
        )

    def validate(self, plantuml_text: str) -> tuple[bool, str]:
        result = self.check(plantuml_text)
        return result.valid, result.message

    @staticmethod
    def _regex_validate(plantuml_text: str) -> tuple[bool, str]:
        """Built-in checks for when PlantUML is not installed.

        Works line by line: block keywords are only recognised at the start
        of a line, so labels such as ":Determine if ...;" are not mistaken
        for an "if" block.
        """
        text = plantuml_text.strip()
        if not text.startswith("@startuml"):
            return False, "Missing @startuml at the beginning"
        if not text.endswith("@enduml"):
            return False, "Missing @enduml at the end"
        if not PlantUMLGenerator._blocks_balanced(text):
            return False, "Unbalanced if/repeat/while/fork blocks"
        for number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if line.startswith(":") and not line.endswith(";"):
                return False, f"line {number}: action must end with ';': {line}"
            if re.match(r"(else)?if\s*\(\s*\)", line):
                return False, f"line {number}: empty if() condition"
        return True, "Built-in validation passed"
