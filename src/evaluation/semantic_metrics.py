from __future__ import annotations

import re
from difflib import SequenceMatcher


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def label_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    sequence = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return 0.5 * jaccard + 0.5 * sequence


def average_best_label_similarity(generated, gold):
    if not generated.nodes or not gold.nodes:
        return 0.0
    scores = []
    gold_labels = [n.label for n in gold.nodes if n.label]
    for n in generated.nodes:
        if not n.label:
            continue
        scores.append(max((label_similarity(n.label, g) for g in gold_labels), default=0.0))
    return sum(scores) / len(scores) if scores else 0.0
