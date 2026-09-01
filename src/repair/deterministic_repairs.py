from __future__ import annotations

from copy import deepcopy

from src.models.domain import (
    ActivityDiagram, ActivityEdge, ActivityNode, Defect, NodeType, RepairResult
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
                ),
            )
            d.edges.insert(
                0,
                ActivityEdge(
                    id="E_INIT",
                    source="N_INIT",
                    target=first.id,
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
                d.nodes.append(ActivityNode(id="N_FINAL", type=NodeType.FINAL, label="End"))
                d.edges.append(ActivityEdge(id="E_FINAL", source=target.id, target="N_FINAL"))
                changes.append("Added missing final node and termination edge.")

        return RepairResult(
            changed=bool(changes),
            repair_type=self.repair_type,
            changes=changes,
            rationale="Applied conservative structural repairs.",
            diagram=d,
        )


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
                            guard="[otherwise]",
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
