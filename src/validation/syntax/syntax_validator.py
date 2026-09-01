from __future__ import annotations

from src.models.domain import (
    ActivityDiagram,
    Defect,
    Severity,
    ValidationResult,
)


class SyntaxValidator:

    def validate(
        self,
        diagram: ActivityDiagram,
    ) -> ValidationResult:

        defects: list[Defect] = []

        try:
            # Force Pydantic validation/serialization of the IR.
            diagram.model_dump()

        except Exception as exc:

            defects.append(
                Defect(
                    id="D-SYN-001",
                    category="SYNTAX",
                    severity=Severity.CRITICAL,
                    description=(
                        "Activity Diagram IR "
                        "cannot be serialized."
                    ),
                    node_ids=[],
                    edge_ids=[],
                    requirement_ids=[],
                    evidence=str(exc),
                    suggested_action=(
                        "Repair the Activity Diagram "
                        "intermediate representation."
                    ),
                )
            )

        return ValidationResult(
            passed=not defects,
            score=(
                1.0
                if not defects
                else 0.0
            ),
            defects=defects,
            rule_counts={
                "SYNTAX-001": len(defects),
            },
            summary=(
                "Activity Diagram IR "
                "syntax validation completed."
            ),
        )