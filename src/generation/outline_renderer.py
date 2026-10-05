"""Draws an ActivityOutline as PlantUML blocks, the way an expert writes it."""
from __future__ import annotations

from src.generation.plantuml_generator import PlantUMLGenerator
from src.models.outline import ActivityOutline, OutlineStep, iter_steps

_escape = PlantUMLGenerator._escape
_escape_guard = PlantUMLGenerator._escape_guard
_escape_lane = PlantUMLGenerator._escape_lane


class OutlineRenderer:
    LANE_TITLE_WIDTH = PlantUMLGenerator._LANE_TITLE_WIDTH

    def render(self, outline: ActivityOutline) -> str:
        lanes = {" ".join(s.lane.split()).casefold() for s in iter_steps(outline.steps) if s.lane}
        self._multi_lane = len(lanes) > 1
        self._names: dict[str, str] = {}
        self._current: str | None = None
        lines = ["@startuml", f"title {_escape(outline.title)}"]
        if self._multi_lane:
            lines.append(f"skinparam swimlaneWrapTitleWidth {self.LANE_TITLE_WIDTH}")
        lines.append("")
        first = next((s.lane for s in iter_steps(outline.steps) if s.lane), None)
        self._lane(first, lines)
        lines.append("start")
        if not self._sequence(outline.steps, lines, depth=0):
            lines.append("stop")
        lines += ["", "@enduml"]
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _name(self, lane: str | None) -> str:
        lane = " ".join((lane or "").split())
        return self._names.setdefault(lane.casefold(), lane) if lane else ""

    def _lane(self, lane: str | None, lines: list[str], force: bool = False) -> None:
        name = self._name(lane)
        if not name or not self._multi_lane:
            return
        if force or name != self._current:
            lines.append(f"|{_escape_lane(name)}|")
            self._current = name

    def _sequence(self, steps: list[OutlineStep], lines: list[str], depth: int) -> bool:
        """Draws the steps; True when the sequence ends in a stop."""
        pad = "  " * depth
        for step in steps:
            if step.kind == "stop":
                if step.label and step.label.lower() not in ("end", "stop"):
                    lines.append(f"{pad}:{_escape(step.label)};")
                lines.append(f"{pad}stop")
                return True
            self._lane(step.lane, lines)
            if step.kind == "action":
                lines.append(f"{pad}:{_escape(step.label)};")
            elif step.kind == "decision":
                if self._decision(step, lines, depth):
                    return True
            elif step.kind == "loop":
                body = step.branches[0] if step.branches else None
                lines.append(f"{pad}while ({_escape(step.label)}) is ({_escape_guard(body.guard if body and body.guard else 'yes')})")
                loop_lane = self._current
                self._sequence(body.steps if body else [], lines, depth + 1)
                lines.append(f"{pad}endwhile ({_escape_guard(step.exit_guard or 'no')})")
                self._lane(loop_lane, lines, force=True)
            elif step.kind == "fork":
                if len(step.branches) < 2:
                    if step.branches and self._sequence(step.branches[0].steps, lines, depth):
                        return True
                    continue
                fork_lane = self._current
                lines.append(f"{pad}fork")
                for index, branch in enumerate(step.branches):
                    if index:
                        lines.append(f"{pad}fork again")
                        self._lane(fork_lane, lines, force=True)
                    self._sequence(branch.steps, lines, depth + 1)
                lines.append(f"{pad}end fork")
                self._lane(fork_lane, lines, force=True)
        return False

    def _decision(self, step: OutlineStep, lines: list[str], depth: int) -> bool:
        """if/else blocks; more than two outcomes nest (PlantUML drops arrows
        of cross-lane elseif chains). True when every branch stops."""
        pad = "  " * depth
        branches = step.branches or []
        if len(branches) < 2:
            branches = list(branches) + [type(branches[0])(guard="no")] if branches else []
        if not branches:
            return False
        decision_lane = self._current
        label = _escape(step.label or "Decision?")
        all_stop = True
        opened = 0
        for index, branch in enumerate(branches):
            guard = _escape_guard(branch.guard or f"option {index + 1}")
            last = index == len(branches) - 1
            if index == 0:
                lines.append(f"{pad}if ({label}) then ({guard})")
                opened += 1
            elif last:
                lines.append(f"{pad}else ({guard})")
            else:
                lines.append(f"{pad}else")
                self._lane(decision_lane, lines, force=True)
                lines.append(f"{pad}if ({label}) then ({guard})")
                opened += 1
            self._lane(decision_lane, lines, force=True)
            stopped = self._sequence(branch.steps, lines, depth + 1)
            all_stop = all_stop and stopped
        lines.extend([f"{pad}endif"] * opened)
        if all_stop:
            return True
        return False
