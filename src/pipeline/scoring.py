from __future__ import annotations

from src.models.domain import PipelineState


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
}


def defect_penalty(defects) -> float:

    penalty = 0.0

    for defect in defects:

        penalty += CATEGORY_WEIGHTS.get(
            defect.category,
            5.0,
        )

        if defect.severity.value == "CRITICAL":
            penalty += 10.0

        elif defect.severity.value == "HIGH":
            penalty += 5.0

        elif defect.severity.value == "MEDIUM":
            penalty += 2.0

    return penalty


def candidate_score(
    validation,
    review,
) -> float:

    validation_score = (
        validation.score
        if validation
        else 0.0
    )

    semantic_score = (
        review.semantic_score
        if review
        else 0.0
    )

    penalty = defect_penalty(
        (
            validation.defects
            if validation
            else []
        )
        +
        (
            review.defects
            if review
            else []
        )
    )

    return (
        validation_score * 50.0
        + semantic_score * 50.0
        - penalty
    )