from __future__ import annotations

import networkx as nx

from src.models.domain import ActivityDiagram


def generate_traces(diagram: ActivityDiagram, max_traces: int = 100, max_depth: int = 30):
    g = nx.MultiDiGraph()
    for n in diagram.nodes:
        g.add_node(n.id, label=n.label, type=n.type.value)
    for e in diagram.edges:
        g.add_edge(e.source, e.target, id=e.id, guard=e.guard)

    initials = [n.id for n in diagram.nodes if n.type.value == "initial"]
    finals = {n.id for n in diagram.nodes if n.type.value == "final"}
    traces = []

    def dfs(node, path, depth):
        if len(traces) >= max_traces or depth > max_depth:
            return
        if node in finals:
            traces.append(path.copy())
            return
        for _, target, data in g.out_edges(node, data=True):
            label = g.nodes[target].get("label") or g.nodes[target].get("type")
            dfs(target, path + [label], depth + 1)

    for start in initials:
        start_label = g.nodes[start].get("label") or g.nodes[start].get("type")
        dfs(start, [start_label], 0)

    return traces
