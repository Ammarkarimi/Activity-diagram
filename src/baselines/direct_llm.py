from __future__ import annotations

from src.agents.requirement_agent import RequirementAgent
from src.agents.generator_agent import GeneratorAgent
from src.llm.openai_client import OpenAIClient
from src.models.domain import ActivityDiagram


def run(requirement_text: str, model: str | None = None) -> ActivityDiagram:
    llm = OpenAIClient(model=model)
    req_agent = RequirementAgent(llm)
    gen = GeneratorAgent(llm)
    extracted = req_agent.run(requirement_text)
    # Direct-ish baseline: requirement extraction is used only to create a compact prompt.
    from src.models.domain import ActivityPlan
    plan = ActivityPlan(
        objective="Direct generation baseline",
        nodes=[],
        edges=[],
        traceability=extracted.matrix,
    )
    return gen.run(requirement_text, extracted.requirements, plan)
