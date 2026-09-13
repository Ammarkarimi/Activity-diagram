from __future__ import annotations

from src.llm.openai_client import OpenAIClient
from src.models.domain import (
    ActivityDiagram,
    Requirement,
    RequirementMatrix,
    ReviewResult,
)


class ReviewerAgent:
    """LLM-based semantic reviewer.

    The prompt uses a private verification checklist rather than requesting
    hidden chain-of-thought. The model returns only the structured review.
    """

    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(
        self,
        requirement_text: str,
        requirements: list[Requirement],
        matrix: RequirementMatrix,
        diagram: ActivityDiagram,
        existing_defects=None,
    ) -> ReviewResult:
        existing_defects = existing_defects or []

        system = """
You are a strict senior UML Activity Diagram semantic reviewer.

PURPOSE
Review the CURRENT diagram against the ORIGINAL requirements. Do not redesign
it and do not reward stylistic preferences.

PRIVATE REASONING / VERIFICATION CHECKLIST
Before producing the structured answer, silently check in this order:
1. Identify the required behaviors, decisions, loops, exceptions,
   concurrency and termination conditions from the atomic requirements.
2. Trace each requirement to the diagram nodes and edges.
3. Follow important execution paths, including negative and retry paths.
4. Check whether the diagram performs any behavior not supported by a
   requirement.
5. Separate true defects from optional modeling improvements.
6. Compare the result with PREVIOUSLY DETECTED DEFECTS so the same logical
   defect is described consistently across iterations.
7. Report only evidence-backed defects.

Do not expose this internal reasoning or a step-by-step chain of thought.
Return only the requested structured ReviewResult.

REVIEW RULES
- Do not report cosmetic/layout issues unless they change meaning.
- Do not report deterministic structural issues already caught by validators,
  unless there is an additional semantic consequence.
- Do not invent requirements or behaviors.
- Do not suggest optional improvements as defects.
- Do not change a defect simply because another valid modeling style exists.
- A defect requires clear evidence that required behavior is missing or that
  generated behavior conflicts with the requirements.
- Keep defect categories stable.
- Prefer the same description, requirement IDs, node IDs and edge IDs for the
  same logical defect so iteration-to-iteration comparison is meaningful.

ALLOWED CATEGORIES
REQUIREMENT_COVERAGE, SEMANTIC, CONTROL_FLOW, DECISION, CONCURRENCY,
EXCEPTION, TERMINATION, HALLUCINATION, GRANULARITY

For every defect provide:
- precise description
- affected node IDs
- affected edge IDs
- affected requirement IDs
- evidence from the requirements/diagram
- concrete repair recommendation

The goal is convergence: a later iteration should have the same or fewer
real defects, not a continuously changing list of subjective suggestions.
"""

        user = f"""
ORIGINAL REQUIREMENTS
=====================
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

Review only the current diagram and return the structured ReviewResult.
"""

        return self.llm.complete(
            system=system,
            user=user,
            response_model=ReviewResult,
        )
