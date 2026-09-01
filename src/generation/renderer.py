from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class PlantUMLRenderer:
    def find_plantuml(self) -> str | None:
        return shutil.which("plantuml")

    def render(self, plantuml_text: str, output_dir: str | Path, name: str = "diagram"):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        source = output / f"{name}.puml"
        source.write_text(plantuml_text, encoding="utf-8")

        exe = self.find_plantuml()
        if not exe:
            return {"source": str(source), "rendered": False}

        proc = subprocess.run(
            [exe, "-tpng", str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        png = output / f"{name}.png"
        return {
            "source": str(source),
            "rendered": proc.returncode == 0 and png.exists(),
            "png": str(png) if png.exists() else None,
            "stderr": proc.stderr,
        }
