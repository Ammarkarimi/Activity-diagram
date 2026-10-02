from __future__ import annotations

from pathlib import Path

from src.generation.plantuml_tool import plantuml_command, run_plantuml


class PlantUMLRenderer:
    def find_plantuml(self) -> list[str] | None:
        return plantuml_command()

    def render(self, plantuml_text: str, output_dir: str | Path, name: str = "diagram"):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        source = output / f"{name}.puml"
        source.write_text(plantuml_text, encoding="utf-8")

        if not self.find_plantuml():
            return {"source": str(source), "rendered": False}

        # SVG has no size limit and opens in any browser; the PNG limit is
        # raised so large diagrams are not cropped.
        stderr = []
        for fmt in ("svg", "png"):
            proc = run_plantuml([f"-t{fmt}", str(source)])
            stderr.append(proc.stderr)
        png = output / f"{name}.png"
        svg = output / f"{name}.svg"
        return {
            "source": str(source),
            "rendered": proc.returncode == 0 and png.exists(),
            "png": str(png) if png.exists() else None,
            "svg": str(svg) if svg.exists() else None,
            "stderr": "".join(stderr),
        }
