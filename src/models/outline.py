"""Block-structured activity outline.

An expert writes an activity diagram as nested blocks: steps in order,
if/else, while loops and forks, each step in the lane of its actor. The
outline keeps exactly that shape, so a diagram built from it never needs
jumps between distant parts of the flow.

The outline is converted to the canonical ActivityDiagram IR for validation
and review: every outline item becomes the node with the same ID (N1, N2,
...), so defects found on the IR point back at outline items.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.models.domain import (
    NON_BEHAVIOURAL_TYPES,
    ActivityDiagram,
    ActivityEdge,
    ActivityNode,
    Defect,
    NodeType,
    Requirement,
    RequirementType,
    Severity,
)

StepKind = Literal["action", "decision", "loop", "fork", "stop"]


class OutlineBranch(BaseModel):
    # decision: the outcome ("yes", "rejected"); loop: the guard under which
    # the body runs again ("no"); fork: empty.
    guard: str = ""
    steps: list["OutlineStep"] = Field(default_factory=list)


class OutlineStep(BaseModel):
    id: str = ""
    kind: StepKind
    # action: short verb phrase; decision/loop: the question; stop: optional
    # outcome ("Request rejected"), fork: empty.
    label: str = ""
    lane: str | None = None
    requirement_ids: list[str] = Field(default_factory=list)
    # decision: one branch per outcome (an empty branch continues after the
    # decision); loop: exactly one branch, the body; fork: one per parallel path.
    branches: list[OutlineBranch] = Field(default_factory=list)
    # loop only: guard on the way out of the loop ("yes").
    exit_guard: str | None = None


OutlineBranch.model_rebuild()


class NotDrawn(BaseModel):
    requirement_id: str
    # DUPLICATE, DETAIL or CONTEXT, with the step it belongs to if any.
    reason: str


class UseCaseNote(BaseModel):
    name: str
    # "first", "follows" (a later phase of the same lifecycle) or
    # "alternative" (another request/variant of the previous use case,
    # drawn as another branch of the same decision).
    relation: str


class ActivityOutline(BaseModel):
    title: str
    # Analysis written BEFORE the steps (the fields are filled in order).
    use_cases: list[UseCaseNote] = Field(default_factory=list)
    shared_steps: list[str] = Field(default_factory=list)
    loops: list[str] = Field(default_factory=list)
    parallel_activities: list[str] = Field(default_factory=list)
    steps: list[OutlineStep] = Field(default_factory=list)
    not_drawn: list[NotDrawn] = Field(default_factory=list)


# ============================================================
# NORMALISATION
# ============================================================


def iter_steps(steps: list[OutlineStep]):
    """Every step, depth first, in drawing order."""
    for step in steps:
        yield step
        for branch in step.branches:
            yield from iter_steps(branch.steps)


def normalize_outline(outline: ActivityOutline) -> ActivityOutline:
    """Renumber steps N2, N3, ... in drawing order (N1 is the initial node),
    drop empty labels on actions, and fill missing lanes from the step before.
    """
    outline = outline.model_copy(deep=True)
    counter = 2
    last_lane: list[str | None] = [None]

    def visit(steps: list[OutlineStep]) -> list[OutlineStep]:
        nonlocal counter
        result = []
        for step in steps:
            if step.kind == "action" and not step.label.strip():
                continue
            step.id = f"N{counter}"
            counter += 1
            step.label = " ".join(step.label.split())
            if step.lane and step.lane.strip():
                step.lane = " ".join(step.lane.split())
                last_lane[0] = step.lane
            elif step.kind != "stop":
                step.lane = last_lane[0]
            if step.kind in ("action", "stop"):
                step.branches = []
            if step.kind == "loop" and len(step.branches) > 1:
                merged = OutlineBranch(guard=step.branches[0].guard)
                for branch in step.branches:
                    merged.steps.extend(branch.steps)
                step.branches = [merged]
            for branch in step.branches:
                lane_before = last_lane[0]
                branch.guard = " ".join((branch.guard or "").split())
                branch.steps = visit(branch.steps)
                last_lane[0] = lane_before
            result.append(step)
            if step.kind == "stop":
                break  # nothing after a stop in the same sequence is reachable
        return result

    outline.steps = visit(outline.steps)
    return outline


def terminates(steps: list[OutlineStep]) -> bool:
    """True when every path through ``steps`` ends in a stop."""
    for step in steps:
        if step.kind == "stop":
            return True
        if step.kind == "decision" and step.branches and all(terminates(b.steps) for b in step.branches):
            return True
    return False


# ============================================================
# OUTLINE -> IR
# ============================================================


def outline_to_diagram(outline: ActivityOutline) -> ActivityDiagram:
    """Converts a (normalised) outline into the ActivityDiagram IR."""
    nodes: list[ActivityNode] = []
    edges: list[ActivityEdge] = []

    def edge(source: str, target: str, guard: str | None = None) -> None:
        edges.append(ActivityEdge(
            id=f"E{len(edges) + 1}", source=source, target=target, guard=guard or None,
        ))

    def node(node_id: str, kind: NodeType, label: str, lane: str | None, reqs: list[str]) -> None:
        nodes.append(ActivityNode(id=node_id, type=kind, label=label, lane=lane, requirement_ids=list(reqs)))

    Exit = tuple[str, str | None]  # (source node, guard of the edge leaving it)

    def connect(exits: list[Exit], target: str) -> None:
        for source, guard in exits:
            edge(source, target, guard)

    def sequence(steps: list[OutlineStep], entry: list[Exit]) -> list[Exit]:
        exits = entry
        for step in steps:
            exits = item(step, exits)
        return exits

    def merged(exits: list[Exit], anchor: OutlineStep) -> list[Exit]:
        # Several ends continue to one next step through a merge node.
        if len(exits) < 2:
            return exits
        merge_id = f"{anchor.id}_M"
        node(merge_id, NodeType.MERGE, "Merge", anchor.lane, [])
        connect(exits, merge_id)
        return [(merge_id, None)]

    def item(step: OutlineStep, entry: list[Exit]) -> list[Exit]:
        if step.kind == "stop":
            node(step.id, NodeType.FINAL, step.label or "End", None, step.requirement_ids)
            connect(entry, step.id)
            return []
        if step.kind == "action":
            node(step.id, NodeType.ACTION, step.label, step.lane, step.requirement_ids)
            connect(entry, step.id)
            return [(step.id, None)]
        if step.kind == "decision":
            node(step.id, NodeType.DECISION, step.label or "Decision?", step.lane, step.requirement_ids)
            connect(entry, step.id)
            exits: list[Exit] = []
            branches = list(step.branches or [OutlineBranch(guard="yes")])
            # Two outcomes that both just continue are one edge, not two.
            empty = [b for b in branches if not b.steps]
            if len(empty) > 1:
                joined = OutlineBranch(guard=" / ".join(b.guard for b in empty if b.guard))
                branches = [b for b in branches if b.steps] + [joined]
            for index, branch in enumerate(branches):
                guard = branch.guard or f"option {index + 1}"
                exits.extend(sequence(branch.steps, [(step.id, guard)]))
            return merged(exits, step)
        if step.kind == "loop":
            node(step.id, NodeType.DECISION, step.label or "Repeat?", step.lane, step.requirement_ids)
            connect(entry, step.id)
            body = step.branches[0] if step.branches else OutlineBranch()
            body_exits = sequence(body.steps, [(step.id, body.guard or "yes")])
            connect(body_exits, step.id)
            return [(step.id, step.exit_guard or "no")]
        if step.kind == "fork":
            if len(step.branches) < 2:
                # A fork with one path is just that path.
                return sequence(step.branches[0].steps if step.branches else [], entry)
            node(step.id, NodeType.FORK, "Fork", step.lane, step.requirement_ids)
            connect(entry, step.id)
            join_id = f"{step.id}_J"
            ends: list[Exit] = []
            for branch in step.branches:
                ends.extend(sequence(branch.steps, [(step.id, None)]))
            node(join_id, NodeType.JOIN, "Join", step.lane, [])
            connect(ends, join_id)
            return [(join_id, None)]
        raise ValueError(f"Unknown outline step kind: {step.kind}")

    node("N1", NodeType.INITIAL, "Start", None, [])
    exits = sequence(outline.steps, [("N1", None)])
    if exits:
        node("N_END", NodeType.FINAL, "End", None, [])
        connect(exits, "N_END")
    return ActivityDiagram(title=outline.title or "Activity Diagram", nodes=nodes, edges=edges)


def copy_lanes(outline: ActivityOutline, diagram: ActivityDiagram) -> None:
    """Gives every outline step the lane of its IR node (same IDs)."""
    lanes = {n.id: n.lane for n in diagram.nodes}
    for step in iter_steps(outline.steps):
        if step.id in lanes and step.kind != "stop":
            step.lane = lanes[step.id]


def traced_requirement_ids(outline: ActivityOutline) -> set[str]:
    return {rid for step in iter_steps(outline.steps) for rid in step.requirement_ids}


def requirements_for_outline(
    requirements: list[Requirement],
    outline: ActivityOutline,
) -> tuple[list[Requirement], list[str]]:
    """Requirements with the ones the outline deliberately leaves out set to OTHER.

    A not_drawn entry counts only for a requirement no step traces, and a
    "DUPLICATE/DETAIL of Rn" only when Rn is drawn. Returns the copied list
    and the IDs set aside.
    """
    traced = traced_requirement_ids(outline)
    set_aside: set[str] = set()
    for entry in outline.not_drawn:
        rid = entry.requirement_id.strip()
        if not rid or rid in traced:
            continue
        reason = entry.reason.strip().upper()
        if reason.startswith(("DUPLICATE", "DETAIL")):
            target = next((w.strip(".,;:()") for w in reason.split()[1:] if w.strip(".,;:()").startswith("R")), None)
            if target not in traced:
                continue
        elif not reason.startswith("CONTEXT"):
            continue
        set_aside.add(rid)
    result = []
    for requirement in requirements:
        copy = requirement.model_copy(deep=True)
        if copy.id in set_aside and copy.type not in NON_BEHAVIOURAL_TYPES:
            copy.type = RequirementType.OTHER
        result.append(copy)
    return result, sorted(set_aside)


def outline_listing(outline: ActivityOutline) -> str:
    """Indented text of the outline with step IDs, lanes and traces (for review)."""
    lines = [f"title: {outline.title}", "N1 start"]

    def visit(steps: list[OutlineStep], depth: int) -> None:
        pad = "  " * depth
        for step in steps:
            reqs = f"  {{{', '.join(step.requirement_ids)}}}" if step.requirement_ids else ""
            lane = f"[{step.lane}] " if step.lane else ""
            if step.kind == "action":
                lines.append(f"{pad}{step.id} {lane}{step.label}{reqs}")
            elif step.kind == "stop":
                lines.append(f"{pad}{step.id} stop{(' (' + step.label + ')') if step.label else ''}")
            elif step.kind == "decision":
                lines.append(f"{pad}{step.id} {lane}DECISION {step.label}{reqs}")
                for branch in step.branches:
                    lines.append(f"{pad}  [{branch.guard}]" + ("" if branch.steps else " -> continue"))
                    visit(branch.steps, depth + 2)
            elif step.kind == "loop":
                body = step.branches[0] if step.branches else OutlineBranch()
                lines.append(f"{pad}{step.id} {lane}LOOP while {step.label} is [{body.guard}]{reqs}")
                visit(body.steps, depth + 1)
                lines.append(f"{pad}end loop, exit when [{step.exit_guard}]")
            elif step.kind == "fork":
                lines.append(f"{pad}{step.id} {lane}FORK")
                for index, branch in enumerate(step.branches, 1):
                    lines.append(f"{pad}  parallel path {index}")
                    visit(branch.steps, depth + 2)
                lines.append(f"{pad}JOIN")

    visit(outline.steps, 0)
    if not outline.steps or not terminates(outline.steps):
        lines.append("stop")
    if outline.not_drawn:
        lines.append("not drawn: " + "; ".join(f"{n.requirement_id} ({n.reason})" for n in outline.not_drawn))
    return "\n".join(lines)


# ============================================================
# OUTLINE CHECKS
# ============================================================


def _key(label: str) -> str:
    return " ".join(label.casefold().split())


def outline_defects(outline: ActivityOutline) -> list[Defect]:
    """Modelling defects visible only in the block structure.

    - a stop inside a parallel path ends every path (blocking);
    - two outcomes of one decision with the same guard (blocking);
    - the same action (label and lane) drawn more than once: shared steps
      belong before or after the branches (not blocking: an expert sometimes
      repeats a step in two alternatives).
    """
    defects: list[Defect] = []

    def has_stop(steps: list[OutlineStep]) -> bool:
        return any(s.kind == "stop" or has_stop([x for b in s.branches for x in b.steps]) for s in steps)

    seen: dict[tuple[str, str], list[str]] = {}
    for step in iter_steps(outline.steps):
        if step.kind == "fork":
            stopping = [i + 1 for i, b in enumerate(step.branches) if has_stop(b.steps)]
            if stopping:
                defects.append(Defect(
                    id=f"D-CON-OUT-STOP-{step.id}", category="CONCURRENCY", severity=Severity.HIGH,
                    description=f"Fork {step.id} has a stop inside parallel path(s) {stopping}; a stop ends "
                                "all parallel paths. Alternatives are a decision, not a fork.",
                    node_ids=[step.id],
                    suggested_action="Turn the fork into a decision if its paths are alternatives, or move "
                                     "the stop after the join.",
                ))
        if step.kind == "decision":
            guards = [_key(b.guard) for b in step.branches if b.guard.strip()]
            if len(guards) != len(set(guards)):
                defects.append(Defect(
                    id=f"D-DECISION-OUT-GUARD-{step.id}", category="DECISION", severity=Severity.HIGH,
                    description=f"Decision {step.id} '{step.label}' has two outcomes with the same guard.",
                    node_ids=[step.id],
                    suggested_action="Give each outcome its own guard and merge outcomes that are the same.",
                ))
        if step.kind == "action":
            seen.setdefault((_key(step.label), _key(step.lane or "")), []).append(step.id)
    kinds = {step.kind for step in iter_steps(outline.steps)}
    if outline.loops and "loop" not in kinds:
        defects.append(Defect(
            id="D-FLOW-OUT-LOOP", category="CONTROL_FLOW", severity=Severity.HIGH,
            description="The outline's analysis lists repeated behaviour (" + "; ".join(outline.loops)
                        + ") but no step is a loop.",
            suggested_action="Draw the repeated steps as a loop step (kind 'loop') with the condition as "
                             "its question, or remove the entry from loops if nothing repeats.",
        ))
    if outline.parallel_activities and "fork" not in kinds:
        defects.append(Defect(
            id="D-CON-OUT-FORK", category="CONCURRENCY", severity=Severity.HIGH,
            description="The outline's analysis lists parallel activities ("
                        + "; ".join(outline.parallel_activities) + ") but no step is a fork.",
            suggested_action="Draw the parallel activities as a fork step, or remove the entry from "
                             "parallel_activities if they happen in sequence.",
        ))
    for (label, _), ids in seen.items():
        if len(ids) > 1:
            defects.append(Defect(
                id=f"D-GRAN-DUP-{ids[0]}", category="GRANULARITY", severity=Severity.MEDIUM,
                description=f"Step '{label}' is drawn {len(ids)} times ({', '.join(ids)}).",
                node_ids=ids,
                suggested_action="Draw a step shared by several branches once, before the decision or "
                                 "after the branches rejoin.",
            ))
    return defects


def outline_coverage(outline: ActivityOutline, requirements: list[Requirement]) -> float:
    """Share of behavioural requirements the outline traces or justifiably leaves out."""
    behavioural = [r for r in requirements if r.type not in NON_BEHAVIOURAL_TYPES]
    if not behavioural:
        return 1.0
    adjusted, _ = requirements_for_outline(requirements, outline)
    remaining = {r.id for r in adjusted if r.type not in NON_BEHAVIOURAL_TYPES}
    traced = traced_requirement_ids(outline)
    return sum(1 for r in behavioural if r.id in traced or r.id not in remaining) / len(behavioural)
