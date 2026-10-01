from __future__ import annotations

import re
from collections import defaultdict

from src.models.domain import (
    ActivityDiagram,
    ActivityEdge,
    ActivityNode,
    DecompositionPlan,
    EdgeType,
    NodeType,
)
from src.pipeline.decomposition import END, START

# Final nodes whose label signals an abnormal end (rejection, failure, ...)
# stay terminal when a module diagram is inlined into the full diagram; any
# other final node means "module finished" and continues with the next module.
_ABNORMAL_END = re.compile(
    r"\b(fail(ed|ure)?|reject(ed|ion)?|cancel(l?ed|lation)?|abort(ed)?|error|"
    r"terminat(e|ed|ion)|den(y|ied)|invalid|timeout|timed out|exception|disband(ed)?)\b",
    re.IGNORECASE,
)


def module_node_id(module_id: str) -> str:
    return f"OV_{module_id}"


def build_overview_diagram(plan: DecompositionPlan) -> ActivityDiagram:
    """Turn a normalized decomposition into a top-level activity diagram.

    Each module becomes one action (a call-behaviour action in UML terms).
    A source with several outgoing transitions gets an explicit decision
    node, so every module action has exactly one outgoing edge, which is
    what the full-diagram composer relies on.
    """
    nodes: list[ActivityNode] = [ActivityNode(id="OV_START", type=NodeType.INITIAL, label="Start")]
    for module in plan.modules:
        nodes.append(
            ActivityNode(
                id=module_node_id(module.id),
                type=NodeType.ACTION,
                label=module.name,
                requirement_ids=list(module.requirement_ids),
            )
        )
    nodes.append(ActivityNode(id="OV_END", type=NodeType.FINAL, label="End"))

    def node_for(ref: str) -> str:
        if ref == START:
            return "OV_START"
        if ref == END:
            return "OV_END"
        return module_node_id(ref)

    outgoing: dict[str, list] = defaultdict(list)
    for transition in plan.transitions:
        outgoing[transition.source].append(transition)

    edges: list[ActivityEdge] = []

    def add_edge(source: str, target: str, guard: str | None = None) -> None:
        edges.append(
            ActivityEdge(
                id=f"OV_E{len(edges) + 1}",
                source=source,
                target=target,
                type=EdgeType.CONTROL,
                guard=guard,
            )
        )

    module_names = {m.id: m.name for m in plan.modules}
    for source, transitions in outgoing.items():
        source_node = node_for(source)
        if len(transitions) == 1:
            add_edge(source_node, node_for(transitions[0].target), transitions[0].guard)
            continue
        decision_id = f"OV_D_{source}"
        nodes.append(
            ActivityNode(
                id=decision_id,
                type=NodeType.DECISION,
                label="Select use case" if source == START else f"Outcome of {module_names.get(source, source)}?",
            )
        )
        add_edge(source_node, decision_id)
        used_guards: set[str] = set()
        for index, transition in enumerate(transitions, start=1):
            guard = transition.guard or (
                module_names.get(transition.target, "finish")
                if transition.target != END
                else "done"
            )
            if guard in used_guards:
                guard = f"{guard} ({index})"
            used_guards.add(guard)
            add_edge(decision_id, node_for(transition.target), guard)

    return ActivityDiagram(
        title=f"{plan.system_name} - Overview",
        nodes=nodes,
        edges=edges,
    )


def compose_full_diagram(
    overview: ActivityDiagram,
    module_diagrams: dict[str, ActivityDiagram | None],
    title: str = "Activity Diagram",
) -> ActivityDiagram:
    """Inline every module diagram into the overview diagram.

    ``module_diagrams`` maps module IDs (M1, M2, ...) to their diagrams. A
    module without a diagram stays as a single call-behaviour action.
    """
    nodes: list[ActivityNode] = []
    edges: list[ActivityEdge] = []
    # Overview node -> node that incoming edges must target.
    entry_of: dict[str, list[tuple[str, str | None]]] = {}
    # Edges that must be connected to the overview successor of a module.
    exits_of: dict[str, list[tuple[str, str | None, list[str]]]] = {}

    module_nodes = {
        node.id: node
        for node in overview.nodes
        if node.id.startswith("OV_M") and node.type == NodeType.ACTION
    }

    for node in overview.nodes:
        module_id = node.id[len("OV_"):] if node.id in module_nodes else None
        sub = module_diagrams.get(module_id) if module_id else None
        if sub is None or not _inlinable(sub):
            nodes.append(node.model_copy(deep=True))
            continue

        prefix = f"{module_id}_"
        initial = next(n for n in sub.nodes if n.type == NodeType.INITIAL)
        finals = [n for n in sub.nodes if n.type == NodeType.FINAL]
        normal_finals = {n.id for n in finals if not _ABNORMAL_END.search(n.label or "")}
        if not normal_finals:
            normal_finals = {n.id for n in finals}

        for sub_node in sub.nodes:
            if sub_node.id == initial.id or sub_node.id in normal_finals:
                continue
            copy = sub_node.model_copy(deep=True)
            copy.id = prefix + sub_node.id
            nodes.append(copy)

        entries: list[tuple[str, str | None]] = []
        exits: list[tuple[str, str | None, list[str]]] = []
        for sub_edge in sub.edges:
            if sub_edge.source == initial.id:
                if sub_edge.target in normal_finals:
                    continue
                entries.append((prefix + sub_edge.target, sub_edge.guard))
                continue
            if sub_edge.target in normal_finals:
                exits.append((prefix + sub_edge.source, sub_edge.guard, list(sub_edge.requirement_ids)))
                continue
            copy = sub_edge.model_copy(deep=True)
            copy.id = prefix + sub_edge.id
            copy.source = prefix + sub_edge.source
            copy.target = prefix + sub_edge.target
            edges.append(copy)

        if len(entries) != 1:
            # Keep a single, explicit entry point for the module.
            entry_node = ActivityNode(
                id=f"{prefix}ENTRY",
                type=NodeType.ACTION,
                label=f"Begin {node.label}",
                requirement_ids=[],
            )
            nodes.append(entry_node)
            for target, guard in entries:
                edges.append(
                    ActivityEdge(
                        id=f"{prefix}ENTRY_E{len(edges)}",
                        source=entry_node.id,
                        target=target,
                        guard=guard,
                    )
                )
            entries = [(entry_node.id, None)]
        if not exits:
            exits = [(entries[0][0], None, [])]
        entry_of[node.id] = entries
        exits_of[node.id] = exits

    for edge in overview.edges:
        sources: list[tuple[str, str | None, list[str]]]
        if edge.source in exits_of:
            sources = [(s, g or edge.guard, r) for s, g, r in exits_of[edge.source]]
        else:
            sources = [(edge.source, edge.guard, list(edge.requirement_ids))]

        target = edge.target
        if target in entry_of:
            target = entry_of[target][0][0]

        for index, (source, guard, requirement_ids) in enumerate(sources, start=1):
            edges.append(
                ActivityEdge(
                    id=f"{edge.id}_{index}",
                    source=source,
                    target=target,
                    type=edge.type,
                    guard=guard,
                    requirement_ids=requirement_ids,
                )
            )

    return ActivityDiagram(title=title, nodes=nodes, edges=_merge_parallel_edges(edges))


def _merge_parallel_edges(edges: list[ActivityEdge]) -> list[ActivityEdge]:
    """Collapse edges with the same source and target (combining guards)."""
    merged: dict[tuple[str, str], ActivityEdge] = {}
    for edge in edges:
        key = (edge.source, edge.target)
        existing = merged.get(key)
        if existing is None:
            merged[key] = edge
            continue
        guards = [g for g in (existing.guard, edge.guard) if g]
        existing.guard = " or ".join(dict.fromkeys(guards)) or None
        for rid in edge.requirement_ids:
            if rid not in existing.requirement_ids:
                existing.requirement_ids.append(rid)
    return list(merged.values())


def _inlinable(diagram: ActivityDiagram) -> bool:
    initials = [n for n in diagram.nodes if n.type == NodeType.INITIAL]
    finals = [n for n in diagram.nodes if n.type == NodeType.FINAL]
    return len(initials) == 1 and bool(finals)
