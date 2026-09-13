from __future__ import annotations

from src.models.domain import Defect, Severity


# Weights used for the research-oriented normalized quality score.
QUALITY_WEIGHTS = {
    "requirement_coverage": 0.25,
    "correctness": 0.25,
    "completeness": 0.20,
    "structural_validity": 0.15,
    "unsupported_behavior": 0.10,
    "defect_free": 0.05,
}

CATEGORY_WEIGHTS = {
    "SYNTAX": 10.0,
    "STRUCTURAL": 8.0,
    "DECISION": 8.0,
    "CONTROL_FLOW": 9.0,
    "CONCURRENCY": 9.0,
    "TERMINATION": 9.0,
    "REQUIREMENT_COVERAGE": 10.0,
    "SEMANTIC": 10.0,
    "EXCEPTION": 8.0,
    "HALLUCINATION": 7.0,
    "GRANULARITY": 4.0,
    "DATA_FLOW": 6.0,
    "LAYOUT": 2.0,
}

SEVERITY_WEIGHTS = {
    Severity.CRITICAL: 4.0,
    Severity.HIGH: 3.0,
    Severity.MEDIUM: 2.0,
    Severity.LOW: 1.0,
}


def defect_penalty(defects: list[Defect]) -> float:
    """Return a transparent penalty used for the auxiliary candidate score."""
    return sum(
        CATEGORY_WEIGHTS.get(defect.category, 5.0)
        * SEVERITY_WEIGHTS.get(defect.severity, 1.0)
        for defect in defects
    )


def quality_score(
    *,
    requirement_coverage: float,
    correctness: float,
    completeness: float,
    structural_validity: float,
    unsupported_behaviour_rate: float,
    defect_count: int,
) -> float:
    """Return a normalized 0..1 quality score.

    This is the main research score. It is deliberately independent of
    reviewer wording and of the raw number of graph nodes/edges.
    """
    unsupported_score = 1.0 - max(0.0, min(1.0, unsupported_behaviour_rate))
    defect_free_score = 1.0 / (1.0 + max(0, defect_count))

    score = (
        QUALITY_WEIGHTS["requirement_coverage"] * requirement_coverage
        + QUALITY_WEIGHTS["correctness"] * correctness
        + QUALITY_WEIGHTS["completeness"] * completeness
        + QUALITY_WEIGHTS["structural_validity"] * structural_validity
        + QUALITY_WEIGHTS["unsupported_behavior"] * unsupported_score
        + QUALITY_WEIGHTS["defect_free"] * defect_free_score
    )
    return max(0.0, min(1.0, score))


def candidate_score(
    validation,
    review,
    defects: list[Defect],
) -> float:
    """Return an auxiliary 0..100 score for iteration logs.

    The orchestrator uses ``quality_score`` for candidate selection and
    repair acceptance. This score is kept for backwards-compatible reporting.
    """
    structural = validation.score if validation else 0.0
    semantic = review.semantic_score if review else 0.0
    penalty = defect_penalty(defects)

    raw = 50.0 * structural + 50.0 * semantic - penalty
    return max(0.0, min(100.0, raw))
