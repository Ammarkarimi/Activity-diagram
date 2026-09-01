from __future__ import annotations

from pydantic import BaseModel

from src.llm.openai_client import OpenAIClient
from src.models.domain import ActivityDiagram


class CoTPlan(BaseModel):
    reasoning_summary: list[str]
    diagram: ActivityDiagram


def run(requirement_text: str, model: str | None = None) -> ActivityDiagram:
    llm = OpenAIClient(model=model)
    system = """Generate a UML Activity Diagram using structured step-by-step decomposition.
Return a concise reasoning summary plus the final ActivityDiagram JSON.
Do not expose hidden chain-of-thought; provide only concise decision/action summaries."""
    user = requirement_text
    return llm.complete(system=system, user=user, response_model=CoTPlan).diagram
