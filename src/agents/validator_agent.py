from __future__ import annotations

from src.models.domain import ActivityDiagram, Requirement, ValidationResult
from src.validation.validator import HybridValidator


class ValidatorAgent:
    """Facade for deterministic + semantic validators."""

    def __init__(self, semantic_validator=None) -> None:
        self.hybrid = HybridValidator(semantic_validator=semantic_validator)

    def run(
        self,
        diagram: ActivityDiagram,
        requirements: list[Requirement],
    ) -> ValidationResult:
        return self.hybrid.validate(diagram, requirements)
