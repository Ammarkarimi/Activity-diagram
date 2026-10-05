"""Which parts of a diagram control flow can actually reach.

Nodes that no path from the initial node reaches are never drawn, so the
requirements traced to them are not shown either. Coverage only counts
reachable elements.
"""
from __future__ import annotations

from src.models.domain import ActivityDiagram, NodeType


def reachable_node_ids(diagram: ActivityDiagram) -> set[str]:
    """Nodes reachable from an initial node (all nodes if there is none)."""
    starts = [n.id for n in diagram.nodes if n.type == NodeType.INITIAL]
    if not starts:
        return {n.id for n in diagram.nodes}
    successors: dict[str, list[str]] = {}
    for edge in diagram.edges:
        successors.setdefault(edge.source, []).append(edge.target)
    seen: set[str] = set()
    stack = list(starts)
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(successors.get(node_id, []))
    return seen


def reachable_requirement_ids(diagram: ActivityDiagram) -> set[str]:
    """Requirements traced to a reachable node or to an edge leaving one."""
    reachable = reachable_node_ids(diagram)
    traced: set[str] = set()
    for node in diagram.nodes:
        if node.id in reachable:
            traced.update(node.requirement_ids)
    for edge in diagram.edges:
        if edge.source in reachable:
            traced.update(edge.requirement_ids)
    return traced
