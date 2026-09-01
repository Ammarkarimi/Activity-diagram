from __future__ import annotations

import networkx as nx


def to_graph(diagram):
    g = nx.DiGraph()
    for n in diagram.nodes:
        g.add_node(n.id, type=n.type.value, label=n.label)
    for e in diagram.edges:
        g.add_edge(e.source, e.target, type=e.type.value, guard=e.guard)
    return g


def normalized_graph_edit_similarity(generated, gold):
    g1 = to_graph(generated)
    g2 = to_graph(gold)
    try:
        ged = nx.graph_edit_distance(
            g1,
            g2,
            node_match=lambda a, b: a.get("type") == b.get("type"),
            edge_match=lambda a, b: a.get("type") == b.get("type"),
        )
        ged = float(ged if ged is not None else 0.0)
    except Exception:
        ged = float(abs(g1.number_of_nodes() - g2.number_of_nodes()) + abs(g1.number_of_edges() - g2.number_of_edges()))
    denom = max(
        1.0,
        g1.number_of_nodes() + g1.number_of_edges() + g2.number_of_nodes() + g2.number_of_edges(),
    )
    return max(0.0, 1.0 - ged / denom)
