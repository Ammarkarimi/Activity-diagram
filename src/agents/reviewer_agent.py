from __future__ import annotations

from src.llm.openai_client import OpenAIClient
from src.models.domain import (
    ActivityDiagram,
    Requirement,
    RequirementMatrix,
    ReviewResult,
)


class ReviewerAgent:

    def __init__(
        self,
        llm: OpenAIClient,
    ) -> None:

        self.llm = llm

    def run(
        self,
        requirement_text: str,
        requirements: list[Requirement],
        matrix: RequirementMatrix,
        diagram: ActivityDiagram,
        existing_defects=None,
    ) -> ReviewResult:

        existing_defects = (
            existing_defects
            or []
        )

        system = """
You are a strict senior UML Activity Diagram reviewer.

Your job is NOT to redesign the diagram.

Your job is to identify semantic defects that are actually
supported by the requirements.

IMPORTANT:

1. Do not report purely cosmetic issues.
2. Do not report issues already covered by deterministic
   structural validation unless they are semantic.
3. Do not invent requirements.
4. Do not suggest optional improvements as defects.
5. Do not change a previous defect merely because another
   modeling style is possible.
6. A defect should be reported only when there is clear
   evidence that the generated behavior conflicts with or
   omits required behavior.
7. Preserve stable defect categories.

Allowed categories:

REQUIREMENT_COVERAGE
SEMANTIC
CONTROL_FLOW
DECISION
CONCURRENCY
EXCEPTION
TERMINATION
HALLUCINATION
GRANULARITY

For each defect provide:

- precise description
- affected node IDs
- affected edge IDs
- affected requirement IDs
- evidence from requirements
- concrete repair recommendation

Do not report the same defect twice.

The goal is convergence, not continuous redesign.
"""

        user = f"""
REQUIREMENTS
============

{requirement_text}


ATOMIC REQUIREMENTS
===================

{[r.model_dump() for r in requirements]}


TRACEABILITY MATRIX
===================

{matrix.model_dump()}


CURRENT ACTIVITY DIAGRAM
========================

{diagram.model_dump()}


PREVIOUSLY DETECTED DEFECTS
============================

{[d.model_dump() for d in existing_defects]}


Review only the current diagram.
"""
        
        return self.llm.complete(
            system=system,
            user=user,
            response_model=ReviewResult,
        )