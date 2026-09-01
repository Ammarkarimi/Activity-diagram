from __future__ import annotations

from src.llm.openai_client import OpenAIClient
from src.agents.generator_agent import GeneratorAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.repair_agent import LLMRepairAgent


def run(requirement_text: str, max_rounds: int = 2, model: str | None = None):
    llm = OpenAIClient(model=model)
    gen = GeneratorAgent(llm)
    reviewer = ReviewerAgent(llm)
    repair = LLMRepairAgent(llm)

    # Minimal self-refine analogue; useful as an ablation, not a claim of exact reproduction.
    from src.agents.requirement_agent import RequirementAgent
    req = RequirementAgent(llm).run(requirement_text)
    from src.models.domain import ActivityPlan
    plan = ActivityPlan(objective="Self-refine baseline", traceability=req.matrix)
    diagram = gen.run(requirement_text, req.requirements, plan)
    for _ in range(max_rounds):
        review = reviewer.run(requirement_text, req.requirements, req.matrix, diagram)
        if not review.defects:
            break
        result = repair.run(requirement_text, req.requirements, diagram, review.defects)
        diagram = result.diagram
    return diagram
