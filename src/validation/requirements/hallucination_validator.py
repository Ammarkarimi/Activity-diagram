from __future__ import annotations

from src.models.domain import (
    ActivityDiagram,
    Defect,
    Requirement,
    Severity,
    ValidationResult,
)


class HallucinationValidator:

    def validate(
        self,
        diagram: ActivityDiagram,
        requirements: list[Requirement],
    ) -> ValidationResult:

        valid_requirement_ids = {
            requirement.id
            for requirement in requirements
        }

        candidates = [
            node
            for node in diagram.nodes
            if node.type.value
            in {
                "action",
                "decision",
                "object",
                "note",
            }
        ]

        unsupported: list[Defect] = []

        for node in candidates:

            supported = bool(
                set(node.requirement_ids)
                & valid_requirement_ids
            )

            if not supported:

                unsupported.append(
                    Defect(
                        id=(
                            f"D-HALL-{node.id}"
                        ),
                        category="HALLUCINATION",
                        severity=Severity.MEDIUM,
                        description=(
                            f"Node {node.id} "
                            "has no supporting "
                            "requirement trace."
                        ),
                        node_ids=[
                            node.id
                        ],
                        edge_ids=[],
                        requirement_ids=[],
                        evidence=node.label,
                        suggested_action=(
                            "Remove the unsupported "
                            "behavior or map it to "
                            "an actual requirement."
                        ),
                    )
                )

        rate = (
            0.0
            if not candidates
            else len(unsupported)
            / len(candidates)
        )

        return ValidationResult(
            passed=not unsupported,
            score=1.0 - rate,
            defects=unsupported,
            rule_counts={
                "HALL-001": len(
                    unsupported
                ),
            },
            summary=(
                "Potential unsupported "
                f"behavior: {len(unsupported)} "
                "node(s)."
            ),
        )