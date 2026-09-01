from __future__ import annotations

from src.models.domain import ActivityDiagram, Requirement, ValidationResult
from src.validation.structural.structural_validator import StructuralValidator
from src.validation.requirements.coverage_validator import CoverageValidator
from src.validation.requirements.hallucination_validator import HallucinationValidator


class HybridValidator:
    def __init__(self, semantic_validator=None) -> None:
        self.structural = StructuralValidator()
        self.coverage = CoverageValidator()
        self.hallucination = HallucinationValidator()
        self.semantic_validator = semantic_validator

    def validate(
        self,
        diagram: ActivityDiagram,
        requirements: list[Requirement],
    ) -> ValidationResult:
        structural = self.structural.validate(diagram)
        coverage = self.coverage.validate(diagram, requirements)
        hallucination = self.hallucination.validate(diagram, requirements)

        defects = structural.defects + coverage.defects + hallucination.defects
        rule_counts = {}
        for result in (structural, coverage, hallucination):
            for k, v in result.rule_counts.items():
                rule_counts[k] = rule_counts.get(k, 0) + v

        semantic_score = 1.0
        if self.semantic_validator is not None:
            semantic = self.semantic_validator.validate(diagram, requirements)
            defects.extend(semantic.defects)
            semantic_score = semantic.score

        passed = len(defects) == 0
        score = max(0.0, 1.0 - min(1.0, len(defects) / max(1, len(diagram.nodes) + len(diagram.edges))))
        score = min(score, semantic_score)

        return ValidationResult(
            passed=passed,
            score=score,
            defects=defects,
            rule_counts=rule_counts,
            summary=f"{len(defects)} defect(s) detected.",
        )
