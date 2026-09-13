from __future__ import annotations

import re


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def defect_key(defect) -> str:
    """Stable identity for the same logical defect across iterations.

    Defect IDs are intentionally excluded because an LLM may call the same
    defect D1 in multiple iterations or use a different generated ID.
    """
    nodes = ",".join(sorted(defect.node_ids))
    edges = ",".join(sorted(defect.edge_ids))
    requirements = ",".join(sorted(defect.requirement_ids))

    # Prefer stable graph/requirement locations. This prevents an LLM from
    # turning the same defect into a new defect merely by rephrasing it.
    location = ",".join([nodes, edges, requirements]).strip(",")
    if location:
        return "|".join([defect.category, location])

    return "|".join([defect.category, normalize(defect.description)])


def defect_signatures(defects) -> set[str]:
    return {defect_key(defect) for defect in defects}


def deduplicate_defects(defects):
    result = []
    seen = set()
    for defect in defects:
        key = defect_key(defect)
        if key in seen:
            continue
        seen.add(key)
        result.append(defect)
    return result
