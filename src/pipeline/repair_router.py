from __future__ import annotations

from src.agents.repair_agent import LLMRepairAgent
from src.llm.openai_client import OpenAIClient
from src.models.domain import (
    ActivityDiagram,
    Defect,
    RepairResult,
    Requirement,
)
from src.repair.deterministic_repairs import (
    DecisionRepair,
    StructuralRepair,
)


class RepairRouter:
    """
    Routes detected defects to the appropriate repair strategy.

    Design principle:
        - Use deterministic repairs whenever the defect can be repaired
          without semantic inference.
        - Use the LLM repair agent for semantic/behavioral repairs.
        - Repair only a small prioritized batch at a time.
        - Never ask the LLM to redesign the whole diagram unnecessarily.
    """

    # Lower number = higher priority.
    REPAIR_PRIORITY: dict[str, int] = {
        "SYNTAX": 1,
        "STRUCTURAL": 2,
        "DECISION": 3,
        "CONTROL_FLOW": 4,
        "CONCURRENCY": 5,
        "TERMINATION": 6,
        "EXCEPTION": 7,
        "REQUIREMENT_COVERAGE": 8,
        "SEMANTIC": 9,
        "HALLUCINATION": 10,
        "DATA_FLOW": 11,
        "GRANULARITY": 12,
        "LAYOUT": 13,
    }

    # Defects that can safely be handled with deterministic logic.
    DETERMINISTIC_CATEGORIES = {
        "STRUCTURAL",
        "DECISION",
    }

    def __init__(
        self,
        llm: OpenAIClient,
        max_defects_per_repair: int = 2,
    ) -> None:

        self.llm = llm

        self.max_defects_per_repair = (
            max(1, max_defects_per_repair)
        )

        self.structural_repair = (
            StructuralRepair()
        )

        self.decision_repair = (
            DecisionRepair()
        )

        self.llm_repair = (
            LLMRepairAgent(llm)
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def repair(
        self,
        requirement_text: str,
        requirements: list[Requirement],
        diagram: ActivityDiagram,
        defects: list[Defect],
    ) -> RepairResult:

        if not defects:

            return RepairResult(
                changed=False,
                repair_type="NONE",
                changes=[],
                rationale="No defects supplied.",
                diagram=diagram,
            )

        selected_defects = (
            self.select_defects(
                defects
            )
        )

        categories = {
            defect.category
            for defect in selected_defects
        }

        # --------------------------------------------------------
        # Deterministic structural repair
        # --------------------------------------------------------

        if categories <= {
            "STRUCTURAL"
        }:

            return self.structural_repair.repair(
                diagram,
                selected_defects,
            )

        # --------------------------------------------------------
        # Deterministic decision repair
        # --------------------------------------------------------

        if categories <= {
            "DECISION"
        }:

            return self.decision_repair.repair(
                diagram,
                selected_defects,
            )

        # --------------------------------------------------------
        # Mixed structural + decision defects
        #
        # Apply deterministic repair first.
        # The orchestrator will validate again.
        # --------------------------------------------------------

        if categories.issubset(
            self.DETERMINISTIC_CATEGORIES
        ):

            current = diagram
            changes: list[str] = []
            changed = False

            structural_defects = [
                defect
                for defect in selected_defects
                if defect.category
                == "STRUCTURAL"
            ]

            decision_defects = [
                defect
                for defect in selected_defects
                if defect.category
                == "DECISION"
            ]

            if structural_defects:

                result = (
                    self.structural_repair.repair(
                        current,
                        structural_defects,
                    )
                )

                current = result.diagram

                changes.extend(
                    result.changes
                )

                changed |= result.changed

            if decision_defects:

                result = (
                    self.decision_repair.repair(
                        current,
                        decision_defects,
                    )
                )

                current = result.diagram

                changes.extend(
                    result.changes
                )

                changed |= result.changed

            return RepairResult(
                changed=changed,
                repair_type="DETERMINISTIC",
                changes=changes,
                rationale=(
                    "Applied deterministic "
                    "structural/decision repairs."
                ),
                diagram=current,
            )

        # --------------------------------------------------------
        # LLM semantic/behavioral repair
        # --------------------------------------------------------

        return self.llm_repair.run(
            requirement_text=requirement_text,
            requirements=requirements,
            diagram=diagram,
            defects=selected_defects,
        )

    # ============================================================
    # DEFECT SELECTION
    # ============================================================

    def select_defects(
        self,
        defects: list[Defect],
    ) -> list[Defect]:

        unique = self._deduplicate(
            defects
        )

        ranked = sorted(
            unique,
            key=lambda defect: (
                self.REPAIR_PRIORITY.get(
                    defect.category,
                    99,
                ),
                self._severity_rank(
                    defect.severity.value
                ),
                defect.id,
            ),
        )

        return ranked[
            : self.max_defects_per_repair
        ]

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    @staticmethod
    def _deduplicate(
        defects: list[Defect],
    ) -> list[Defect]:

        result: list[Defect] = []

        seen: set[tuple] = set()

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
                defect.description.strip().lower(),
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(
                defect
            )

        return result

    # ============================================================
    # SEVERITY
    # ============================================================

    @staticmethod
    def _severity_rank(
        severity: str,
    ) -> int:

        return {
            "CRITICAL": 1,
            "HIGH": 2,
            "MEDIUM": 3,
            "LOW": 4,
        }.get(
            severity,
            5,
        )