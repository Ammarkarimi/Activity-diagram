from __future__ import annotations

from src.models.domain import (
    ActivityDiagram,
    Defect,
    Requirement,
    Severity,
    ValidationResult,
)


class CoverageValidator:

    def validate(
        self,
        diagram: ActivityDiagram,
        requirements: list[Requirement],
    ) -> ValidationResult:

        represented: set[str] = set()

        for node in diagram.nodes:
            represented.update(
                node.requirement_ids
            )

        for edge in diagram.edges:
            represented.update(
                edge.requirement_ids
            )

        defects: list[Defect] = []

        for requirement in requirements:

            if requirement.id not in represented:

                defects.append(
                    Defect(
                        id=f"D-COV-{requirement.id}",
                        category="REQUIREMENT_COVERAGE",
                        severity=Severity.HIGH,
                        description=(
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