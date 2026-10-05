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
        "CONTROL_FLOW": 3,
        "TERMINATION": 4,
        # Missing behaviour is repaired before refinements of what is drawn.
        "REQUIREMENT_COVERAGE": 5,
        "DECISION": 6,
        "CONCURRENCY": 7,
        "EXCEPTION": 8,
        "SEMANTIC": 9,
        "HALLUCINATION": 10,
        "DATA_FLOW": 11,
        "GRANULARITY": 12,
        "LAYOUT": 13,
    }

    # Defects the deterministic rules actually fix (missing initial or final
    # node, an action with several exits). Missing guards go to the LLM,
    # which names them; the old rule wrote "branch_1".
    DETERMINISTIC_IDS = ("D-STRUCT-001", "D-STRUCT-002", "D-FLOW-002")


    def __init__(
        self,
        llm: OpenAIClient,
        max_defects_per_repair: int = 3,
    ) -> None:

        self.llm = llm

        self.max_defects_per_repair = (
            max(1, max_defects_per_repair)
        )

        self.structural_repair = (
            StructuralRepair()
        )


        self.llm_repair = (
            LLMRepairAgent(llm)
        )

        # Whether the last repair() call used the LLM (for call counting).
        self.last_used_llm = False

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

        self.last_used_llm = False

        # Rules fix only the defects they are written for; the rest of the
        # batch (unreachable nodes, decisions without effect, ...) is a
        # semantic question for the LLM. Sending a mixed batch to the rules
        # alone left those defects unrepaired, and a batch the rules could
        # not change at all stopped the repair loop.
        fixable = [d for d in selected_defects if d.id.startswith(self.DETERMINISTIC_IDS)]
        rest = [d for d in selected_defects if d not in fixable]

        current = diagram
        changes: list[str] = []
        if fixable:
            result = self.structural_repair.repair(diagram, fixable)
            if result.changed:
                current = result.diagram
                changes = list(result.changes)

        if not rest:
            if changes:
                return RepairResult(
                    changed=True,
                    repair_type="DETERMINISTIC",
                    changes=changes,
                    rationale="Applied deterministic structural repairs.",
                    diagram=current,
                )
            rest = selected_defects

        self.last_used_llm = True
        llm_result = self.llm_repair.run(
            requirement_text=requirement_text,
            requirements=requirements,
            diagram=current,
            defects=rest,
        )
        if not changes:
            return llm_result
        if not llm_result.changed:
            return RepairResult(
                changed=True,
                repair_type="DETERMINISTIC",
                changes=changes,
                rationale="Applied deterministic structural repairs; the LLM repair changed nothing.",
                diagram=current,
            )
        llm_result.changes = changes + list(llm_result.changes)
        return llm_result

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