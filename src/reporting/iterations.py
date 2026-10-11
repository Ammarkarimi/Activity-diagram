"""Iteration-by-iteration view of a run.

For every candidate the pipeline evaluated: the defects verification found,
the ones sent to repair, what the repair changed and whether it was kept,
and which defects the next candidate no longer has. The candidate's PlantUML
is coloured so the steps a defect points at (red) and the steps a repair
added (green) can be seen in the diagram itself.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from src.generation.plantuml_generator import PlantUMLGenerator
from src.models.domain import CandidateRecord, Defect, NodeType, PipelineState

FLAGGED = "#FFCDD2"
ADDED = "#C8E6C9"

_ACTION = re.compile(r"^(\s*)(#[0-9A-Za-z]+)?:(.*);\s*$")
_NODE_ID = re.compile(r"\bN\d+(?:_[A-Z])?\b")


@dataclass
class IterationView:
    iteration: int
    candidate: CandidateRecord
    defects: list[Defect]
    blocking_ids: set[str]
    selected_ids: set[str]
    repairs: list[dict]
    resolved: list[Defect] = field(default_factory=list)
    added_steps: list[str] = field(default_factory=list)
    removed_steps: list[str] = field(default_factory=list)
    plantuml: str = ""


def unique_defects(defects: list[Defect]) -> list[Defect]:
    """One entry per defect ID (the validator and the reviewer may both report it)."""
    seen: dict[str, Defect] = {}
    for defect in defects:
        seen.setdefault(defect.id, defect)
    return list(seen.values())


def _labels(candidate: CandidateRecord) -> dict[str, str]:
    return {n.id: n.label for n in candidate.diagram.nodes}


def _identity(defect: Defect, labels: dict[str, str]) -> str:
    """Same logical defect across candidates, although node IDs are renumbered."""
    if defect.requirement_ids:
        where = ",".join(sorted(defect.requirement_ids))
    elif defect.node_ids:
        where = ",".join(sorted(labels.get(n, n).casefold() for n in defect.node_ids))
    else:
        where = _NODE_ID.sub("N", " ".join(defect.description.lower().split()))
    return f"{defect.category}|{_NODE_ID.sub('N', defect.id)}|{where}"


def _action_nodes(candidate: CandidateRecord):
    """Action nodes in drawing order (outline IDs N2, N3, ... follow it)."""
    def order(node_id: str) -> tuple[int, str]:
        match = re.fullmatch(r"N(\d+)", node_id)
        return (int(match.group(1)), "") if match else (10**9, node_id)
    nodes = [n for n in candidate.diagram.nodes if n.type == NodeType.ACTION]
    return sorted(nodes, key=lambda n: order(n.id))


def highlight(plantuml: str, candidate: CandidateRecord, colours: dict[str, str]) -> str:
    """Colour the action boxes of the given node IDs.

    Actions are matched by label; when a label occurs several times, the
    n-th node with that label (in ID order) is the n-th box with it.
    """
    if not colours:
        return plantuml
    wanted: dict[tuple[str, int], str] = {}
    seen: Counter = Counter()
    for node in _action_nodes(candidate):
        text = PlantUMLGenerator._escape(node.label)
        if node.id in colours:
            wanted[(text, seen[text])] = colours[node.id]
        seen[text] += 1
    lines, drawn = [], Counter()
    for line in plantuml.splitlines():
        match = _ACTION.match(line)
        if match:
            pad, _, text = match.groups()
            colour = wanted.get((text, drawn[text]))
            drawn[text] += 1
            if colour:
                line = f"{pad}{colour}:{text};"
        lines.append(line)
    return "\n".join(lines)


def iteration_views(state: PipelineState, blocking_prefixes: tuple[str, ...]) -> list[IterationView]:
    views: list[IterationView] = []
    previous: CandidateRecord | None = None
    previous_defects: list[Defect] = []
    for candidate in sorted(state.candidates, key=lambda c: c.iteration):
        defects = unique_defects(candidate.defects)
        repairs = [h for h in state.repair_history if h.get("iteration") == candidate.iteration]
        selected = {i for h in repairs for i in h.get("defect_ids", [])}
        view = IterationView(
            iteration=candidate.iteration,
            candidate=candidate,
            defects=defects,
            blocking_ids={d.id for d in defects if d.id.startswith(blocking_prefixes)},
            selected_ids=selected,
            repairs=repairs,
        )
        labels = _labels(candidate)
        colours: dict[str, str] = {}
        for defect in defects:
            if defect.id in selected:
                for node_id in defect.node_ids:
                    colours[node_id] = FLAGGED
        # A step the last repair added stays green even when a later defect names it.
        if previous is not None:
            before_labels = _labels(previous)
            now = {_identity(d, labels) for d in defects}
            view.resolved = [d for d in previous_defects if _identity(d, before_labels) not in now]
            old = Counter(n.label.casefold() for n in _action_nodes(previous))
            new = Counter(n.label.casefold() for n in _action_nodes(candidate))
            view.added_steps = [n.label for n in _action_nodes(candidate) if n.label.casefold() not in old]
            view.removed_steps = [n.label for n in _action_nodes(previous) if n.label.casefold() not in new]
            fixed_requirements = {r for d in view.resolved for r in d.requirement_ids}
            for node in _action_nodes(candidate):
                if node.label.casefold() not in old or set(node.requirement_ids) & fixed_requirements:
                    colours[node.id] = ADDED
        view.plantuml = highlight(candidate.plantuml, candidate, colours)
        views.append(view)
        previous, previous_defects = candidate, defects
    return views
