from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm.openai_client import OpenAIClient
from src.models.domain import Defect


class FeedbackResult(BaseModel):
    prioritized_defects: list[Defect] = Field(default_factory=list)
    final_recommendation: str = ""


class FeedbackAgent:
    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(self, defects: list[Defect]) -> FeedbackResult:
        system = """You are a defect triage specialist.
Prioritize the supplied model defects by impact on semantic and behavioral correctness.
Do not invent defects. Return the same defect IDs where possible."""
        user = f"Defects:\n{[d.model_dump() for d in defects]}"
        return self.llm.complete(
            system=system,
            user=user,
            response_model=FeedbackResult,
        )
