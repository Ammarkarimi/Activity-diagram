from __future__ import annotations

from src.evaluation.behavioral_metrics import behavioral_similarity
from src.evaluation.complexity_metrics import complexity
from src.evaluation.graph_metrics import normalized_graph_edit_similarity
from src.evaluation.requirement_metrics import requirement_coverage, unsupported_behavior_rate
from src.evaluation.semantic_metrics import average_best_label_similarity


def evaluate(generated, gold, requirements):
    out = {
        "requirement_coverage": requirement_coverage(generated, requirements),
        "unsupported_behavior_rate": unsupported_behavior_rate(generated, requirements),
        "graph_similarity": normalized_graph_edit_similarity(generated, gold),
        "label_similarity": average_best_label_similarity(generated, gold),
    }
    out.update(behavioral_similarity(generated, gold))
    out.update({f"complexity_{k}": v for k, v in complexity(generated).items()})
    return out
