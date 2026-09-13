from __future__ import annotations

from src.models.domain import (
    ActivityDiagram,
    Requirement,
    ValidationResult,
)

from src.validation.structural.structural_validator import (
    StructuralValidator,
)

from src.validation.requirements.coverage_validator import (
    CoverageValidator,
)

from src.validation.requirements.hallucination_validator import (
    HallucinationValidator,
)


class HybridValidator:
    """
    Combines deterministic validators into a single validation stage.

    Validation dimensions:

        1. Structural validity
        2. Requirement coverage
        3. Hallucination / unsupported behaviour
        4. Optional semantic validation

    The validator is intentionally deterministic wherever possible.
    Semantic reasoning is delegated to the optional semantic validator.

    The resulting ValidationResult is consumed by the orchestrator for:

        - defect detection
        - candidate scoring
        - repair decisions
        - iteration tracking
        - research evaluation metrics
    """

    def __init__(
        self,
        semantic_validator=None,
    ) -> None:

        self.structural = (
            StructuralValidator()
        )

        self.coverage = (
            CoverageValidator()
        )

        self.hallucination = (
            HallucinationValidator()
        )

        self.semantic_validator = (
            semantic_validator
        )

    # ============================================================
    # MAIN VALIDATION
    # ============================================================

    def validate(
        self,
        diagram: ActivityDiagram,
        requirements: list[Requirement],
    ) -> ValidationResult:

        # --------------------------------------------------------
        # 1. Structural validation
        # --------------------------------------------------------

        structural = (
            self.structural.validate(
                diagram
            )
        )

        # --------------------------------------------------------
        # 2. Requirement coverage validation
        # --------------------------------------------------------

        coverage = (
            self.coverage.validate(
                diagram,
                requirements,
            )
        )

        # --------------------------------------------------------
        # 3. Hallucination validation
        # --------------------------------------------------------

        hallucination = (
            self.hallucination.validate(
                diagram,
                requirements,
            )
        )

        # --------------------------------------------------------
        # 4. Combine deterministic defects
        # --------------------------------------------------------

        defects = (
            structural.defects
            + coverage.defects
            + hallucination.defects
        )

        # --------------------------------------------------------
        # 5. Combine rule counts
        # --------------------------------------------------------

        rule_counts: dict[str, int] = {}

        for result in (
            structural,
            coverage,
            hallucination,
        ):

            for key, value in (
                result.rule_counts.items()
            ):

                rule_counts[key] = (
                    rule_counts.get(
                        key,
                        0,
                    )
                    + value
                )

        # --------------------------------------------------------
        # 6. Individual validation scores
        # --------------------------------------------------------

        structural_score = (
            structural.score
            if structural is not None
            else 0.0
        )

        coverage_score = (
            coverage.score
            if coverage is not None
            else 0.0
        )

        hallucination_score = (
            hallucination.score
            if hallucination is not None
            else 1.0
        )

        # --------------------------------------------------------
        # 7. Optional semantic validation
        # --------------------------------------------------------

        semantic_score = 1.0

        if (
            self.semantic_validator
            is not None
        ):

            semantic = (
                self.semantic_validator.validate(
                    diagram,
                    requirements,
                )
            )

            defects.extend(
                semantic.defects
            )

            semantic_score = (
                semantic.score
            )

        # --------------------------------------------------------
        # 8. Remove duplicate defects
        # --------------------------------------------------------

        defects = (
            self._deduplicate_defects(
                defects
            )
        )

        # --------------------------------------------------------
        # 9. Overall deterministic score
        # --------------------------------------------------------

        total_elements = max(
            1,
            len(diagram.nodes)
            + len(diagram.edges),
        )

        defect_density = (
            len(defects)
            / total_elements
        )

        defect_based_score = max(
            0.0,
            1.0
            - min(
                1.0,
                defect_density,
            ),
        )

        # The overall validation score should represent the
        # weakest important validation dimension rather than
        # hiding a serious failure behind an average.

        deterministic_score = min(
            structural_score,
            coverage_score,
            hallucination_score,
        )

        score = min(
            deterministic_score,
            semantic_score,
            defect_based_score,
        )

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        # --------------------------------------------------------
        # 10. Pass / fail
        # --------------------------------------------------------

        passed = (
            len(defects) == 0
        )

        # --------------------------------------------------------
        # 11. Additional research-oriented rule counts
        # --------------------------------------------------------

        rule_counts = {
            **rule_counts,

            "TOTAL_DEFECTS": (
                len(defects)
            ),

            "TOTAL_NODES": (
                len(diagram.nodes)
            ),

            "TOTAL_EDGES": (
                len(diagram.edges)
            ),

            "TOTAL_REQUIREMENTS": (
                len(requirements)
            ),
        }

        # --------------------------------------------------------
        # 12. Human-readable summary
        # --------------------------------------------------------

        summary = (
            f"{len(defects)} defect(s) detected. "
            f"Structural score: "
            f"{structural_score:.3f}. "
            f"Coverage score: "
            f"{coverage_score:.3f}. "
            f"Hallucination score: "
            f"{hallucination_score:.3f}. "
            f"Semantic score: "
            f"{semantic_score:.3f}. "
            f"Overall score: "
            f"{score:.3f}."
        )

        # --------------------------------------------------------
        # 13. Return unified result
        # --------------------------------------------------------

        return ValidationResult(
            passed=passed,
            score=score,
            defects=defects,
            rule_counts=rule_counts,
            summary=summary,
        )

    # ============================================================
    # DEFECT DEDUPLICATION
    # ============================================================

    @staticmethod
    def _deduplicate_defects(
        defects,
    ):

        result = []

        seen = set()

        for defect in defects:

            key = (
                defect.category,

                tuple(
                    sorted(
                        defect.node_ids
                    )
                ),

                tuple(
                    sorted(
                        defect.edge_ids
                    )
                ),

                tuple(
                    sorted(
                        defect.requirement_ids
                    )
                ),

                defect.description
                .strip()
                .lower(),
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(
                defect
            )

        return result