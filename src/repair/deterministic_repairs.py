from __future__ import annotations

from copy import deepcopy

from src.models.domain import (
    ActivityDiagram, ActivityEdge, ActivityNode, Defect, NodeType, RepairResult, EdgeType
)


class StructuralRepair:
    repair_type = "STRUCTURAL"

    def repair(self, diagram: ActivityDiagram, defects: list[Defect]) -> RepairResult:
        d = deepcopy(diagram)
        changes = []

        initials = [n for n in d.nodes if n.type == NodeType.INITIAL]
        if not initials and d.nodes:
            first = d.nodes[0]
            d.nodes.insert(
                0,
                ActivityNode(
                    id="N_INIT",
                    type=NodeType.INITIAL,
                    label="Start",
                    requirement_ids=[],
                ),
            )
            d.edges.insert(
                0,
                ActivityEdge(
                    id="E_INIT",
                    source="N_INIT",
                    target=first.id,
                    type=EdgeType.CONTROL,
                    guard=None,
                    requirement_ids=first.requirement_ids.copy(),
                ),
            )
            changes.append("Added missing initial node.")

        finals = [n for n in d.nodes if n.type == NodeType.FINAL]
        if not finals and d.nodes:
            terminal_candidates = {
                e.source for e in d.edges
            }
            candidates = [n for n in d.nodes if n.id not in terminal_candidates and n.type != NodeType.INITIAL]
            if candidates:
                target = candidates[-1]
                d.nodes.append(ActivityNode(id="N_FINAL", type=NodeType.FINAL, label="End", requirement_ids=[]))
                d.edges.append(ActivityEdge(id="E_FINAL", source=target.id, target="N_FINAL", type=EdgeType.CONTROL, guard=None, requirement_ids=[]))
                changes.append("Added missing final node and termination edge.")

        changes.extend(self._split_branching_actions(d))

        return RepairResult(
            changed=bool(changes),
            repair_type=self.repair_type,
            changes=changes,
            rationale="Applied conservative structural repairs.",
            diagram=d,
        )

    @staticmethod
    def _split_branching_actions(d: ActivityDiagram) -> list[str]:
        """An action with several outgoing flows hides a decision: insert
        one after the action and move the flows to it (guards kept)."""
        changes = []
        node_ids = {n.id for n in d.nodes}
        edge_ids = {e.id for e in d.edges}
        for node in list(d.nodes):
            if node.type != NodeType.ACTION:
                continue
            outgoing = [e for e in d.edges if e.source == node.id]
            if len(outgoing) < 2:
                continue
            # Guarded flows are a choice; unguarded ones all run (UML implicit
            # fork), so they get a fork node instead of a guardless decision.
            parallel = not any((e.guard or "").strip("[] ") for e in outgoing)
            decision_id = _unique(f"{node.id}_{'F' if parallel else 'D'}", node_ids)
            edge_id = _unique(f"E_{decision_id}", edge_ids)
            d.nodes.insert(
                d.nodes.index(node) + 1,
                ActivityNode(
                    id=decision_id,
                    type=NodeType.FORK if parallel else NodeType.DECISION,
                    label="Fork" if parallel else _decision_label(outgoing),
                    lane=node.lane,
                    requirement_ids=node.requirement_ids.copy(),
                ),
            )
            for e in outgoing:
                e.source = decision_id
            d.edges.append(
                ActivityEdge(
                    id=edge_id,
                    source=node.id,
                    target=decision_id,
                    type=EdgeType.CONTROL,
                    guard=None,
                    requirement_ids=[],
                )
            )
            changes.append(
                f"Inserted {'fork' if parallel else 'decision'} {decision_id} after action {node.id}, "
                f"which had {len(outgoing)} outgoing flows."
            )
        return changes


def _unique(base: str, taken: set[str]) -> str:
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}{n}", n + 1
    taken.add(candidate)
    return candidate


def _decision_label(outgoing: list[ActivityEdge]) -> str:
    """'Unicast, multicast or broadcast?' from the guards of the flows."""
    options = [
        (e.guard or "").strip().strip("[]").strip()
        for e in outgoing
    ]
    options = [o for o in dict.fromkeys(options) if o]
    if not options:
        return "Which path?"
    text = options[0] if len(options) == 1 else ", ".join(options[:-1]) + " or " + options[-1]
    return text[0].upper() + text[1:] + "?"


class DecisionRepair:
    repair_type = "DECISION"

    def repair(self, diagram: ActivityDiagram, defects: list[Defect]) -> RepairResult:
        d = deepcopy(diagram)
        changes = []
        edge_counter = len(d.edges) + 1

        for node in d.nodes:
            if node.type != NodeType.DECISION:
                continue
            outgoing = [e for e in d.edges if e.source == node.id]
            for idx, e in enumerate(outgoing):
                if not e.guard:
                    e.guard = f"branch_{idx + 1}"
                    changes.append(f"Added guard to {e.id}: {e.guard}")
            if len(outgoing) == 1:
                # Conservative fallback: add a self-contained note-like alternative branch
                # only when there is a known merge/final target in the graph.
                target = next((n for n in d.nodes if n.type == NodeType.FINAL), None)
                if target:
                    d.edges.append(
                        ActivityEdge(
                            id=f"E{edge_counter}",
                            source=node.id,
                            target=target.id,
                            type=EdgeType.CONTROL,
                            guard="[otherwise]",
                            requirement_ids=[],
                        )
                    )
                    edge_counter += 1
                    changes.append(f"Added otherwise branch from {node.id}.")
        return RepairResult(
            changed=bool(changes),
            repair_type=self.repair_type,
            changes=changes,
            rationale="Repaired missing decision guards/otherwise path where possible.",
            diagram=d,
        )


class ConcurrencyRepair:
    repair_type = "CONCURRENCY"

    def repair(self, diagram: ActivityDiagram, defects: list[Defect]) -> RepairResult:
        # Avoid inventing concurrency. This repair only adds no behavior and returns unchanged.
        return RepairResult(
            changed=False,
            repair_type=self.repair_type,
            changes=[],
            rationale="Concurrency repairs are intentionally LLM-guided because branch intent is semantic.",
            diagram=diagram,
        )
