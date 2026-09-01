from __future__ import annotations

from src.models.domain import ActivityDiagram, Requirement, ReviewResult
from src.agents.reviewer_agent import ReviewerAgent


class SemanticValidator:
    def __init__(self, reviewer: ReviewerAgent) -> None:
        self.reviewer = reviewer

    def validate(self, diagram: ActivityDiagram, requirements: list[Requirement]):
        result: ReviewResult = self.reviewer.run(
            requirement_text="\n".join(f"{r.id}: {r.text}" for r in requirements),
            requirements=requirements,
            matrix=None,  # reviewer can operate without a pre-built matrix here
            diagram=diagram,
        )
        from src.models.domain import ValidationResult
        return ValidationResult(
            passed=not result.defects,
            score=result.semantic_score,
            defects=result.defects,
            summary=result.overall_assessment,
        )
