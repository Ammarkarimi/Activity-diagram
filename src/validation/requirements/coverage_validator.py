from __future__ import annotations

from src.models.domain import (
    ActivityDiagram,
    Defect,
    Requirement,
    Severity,
    ValidationResult,
    modelled_requirements,
)
from src.validation.reachability import reachable_requirement_ids


class CoverageValidator:

    def validate(
        self,
        diagram: ActivityDiagram,
        requirements: list[Requirement],
    ) -> ValidationResult:

        # Only reachable elements count: an unreachable node is never drawn.
        represented = reachable_requirement_ids(diagram)
        traced_anywhere = {
            rid
            for element in [*diagram.nodes, *diagram.edges]
            for rid in element.requirement_ids
        }

        defects: list[Defect] = []
        requirements = modelled_requirements(requirements)

        for requirement in requirements:

            if requirement.id not in represented:

                defects.append(
                    Defect(
                        id=f"D-COV-{requirement.id}",
                        category="REQUIREMENT_COVERAGE",
                        severity=Severity.HIGH,
                        description=(
                            f"Requirement "
                            f"{requirement.id} is only "
                            "traced to unreachable "
                            "elements, so it is not shown."
                            if requirement.id in traced_anywhere
                            else
                            f"Requirement "
                            f"{requirement.id} is not "
                            "traced to any diagram "
                            "element."
                        ),
                        node_ids=[],
                        edge_ids=[],
                        requirement_ids=[
                            requirement.id
                        ],
                        evidence=requirement.text,
                        suggested_action=(
                            "Add or connect a diagram "
                            "element representing "
                            "this requirement."
                        ),
                    )
                )

        coverage = (
            1.0
            if not requirements
            else (
                len(requirements)
                - len(defects)
            )
            / len(requirements)
        )

        return ValidationResult(
            passed=not defects,
            score=coverage,
            defects=defects,
            rule_counts={
                "COVERAGE-001": len(defects),
            },
            summary=(
                f"Requirement coverage: "
                f"{coverage:.3f}"
            ),
        )