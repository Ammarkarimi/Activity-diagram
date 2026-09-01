from src.models.domain import ActivityDiagram


def validate_ir(diagram: ActivityDiagram) -> None:
    ids = {n.id for n in diagram.nodes}
    if len(ids) != len(diagram.nodes):
        raise ValueError("Duplicate node IDs.")
    edge_ids = {e.id for e in diagram.edges}
    if len(edge_ids) != len(diagram.edges):
        raise ValueError("Duplicate edge IDs.")
    for e in diagram.edges:
        if e.source not in ids or e.target not in ids:
            raise ValueError(f"Edge {e.id} references a missing node.")
