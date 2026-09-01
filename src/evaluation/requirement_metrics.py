from __future__ import annotations


def requirement_coverage(diagram, requirements):
    represented = set()
    for n in diagram.nodes:
        represented.update(n.requirement_ids)
    for e in diagram.edges:
        represented.update(e.requirement_ids)
    return 1.0 if not requirements else sum(r.id in represented for r in requirements) / len(requirements)


def unsupported_behavior_rate(diagram, requirements):
    valid = {r.id for r in requirements}
    candidates = [
        n for n in diagram.nodes
        if n.type.value in {"action", "decision", "object", "note"}
    ]
    if not candidates:
        return 0.0
    unsupported = sum(not (set(n.requirement_ids) & valid) for n in candidates)
    return unsupported / len(candidates)
