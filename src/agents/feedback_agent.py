from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm.openai_client import OpenAIClient
from src.models.domain import Defect


class FeedbackResult(BaseModel):
    prioritized_defects: list[Defect] = Field(default_factory=list)
    final_recommendation: str = ""


class FeedbackAgent:
    """Ranks existing defects; it must never create new ones."""

    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(self, defects: list[Defect]) -> FeedbackResult:
        system = """
You are a defect triage specialist for an iterative UML repair system.

PRIVATE VERIFICATION CHECKLIST
Silently:
1. Compare the supplied defects by severity and behavioral impact.
2. Put blocking correctness/behavior defects before minor defects.
3. Preserve every defect's original ID and details.
4. Do not create, merge, rewrite, or remove defects.
5. Return only a reordered subset of the supplied defects.

Do not expose internal reasoning. Return the structured result only.

The purpose is prioritization, not diagnosis. The orchestrator will verify
that every returned defect already exists in the current defect set.
"""
        user = f"""
CURRENT DEFECTS
==============
{[d.model_dump() for d in defects]}

Prioritize these defects for the next repair batch. Return the same defect
objects, preferably with their original IDs.
"""
        return self.llm.complete(
            system=system,
            user=user,
            response_model=FeedbackResult,
        )
