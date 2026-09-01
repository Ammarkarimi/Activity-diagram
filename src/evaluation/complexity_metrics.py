from __future__ import annotations

import networkx as nx


def complexity(diagram) -> dict:
    g = nx.DiGraph()
    g.add_nodes_from(n.id for n in diagram.nodes)
    g.add_edges_from((e.source, e.target) for e in diagram.edges)

    decisions = sum(n.type.value == "decision" for n in diagram.nodes)
    forks = sum(n.type.value == "fork" for n in diagram.nodes)
    joins = sum(n.type.value == "join" for n in diagram.nodes)
    edges = len(diagram.edges)
    nodes = len(diagram.nodes)

    # McCabe-style graph complexity approximation for directed workflow graphs.
    cyclomatic = edges - nodes + 2 if nodes else 0

    return {
        "nodes": nodes,
        "edges": edges,
        "decisions": decisions,
        "forks": forks,
        "joins": joins,
        "cyclomatic_approx": max(0, cyclomatic),
    }
