from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm.openai_client import OpenAIClient
from src.models.domain import Requirement, RequirementType


class RequirementVerdict(BaseModel):
    id: str
    # STEP: a step of its own. DUPLICATE: restates another requirement's step.
    # DETAIL: how a step is done, or a constraint on it. CONTEXT: no behaviour.
    kind: str
    # For DUPLICATE and DETAIL: the STEP requirement it belongs to.
    belongs_to: str | None = None


class ConsolidationResult(BaseModel):
    verdicts: list[RequirementVerdict] = Field(default_factory=list)


_SYSTEM_PROMPT = """
You are a senior UML modeller preparing a requirement list for ONE activity
diagram. The list was extracted section by section from one specification,
so the same step often appears several times (in an overview, a component
description and a scenario), and some entries only explain how a step works.

Classify EVERY requirement:
- STEP: a step of the behaviour, a decision or condition that changes the
  flow, a loop, a concurrent activity, an exception or a termination.
  Keep the most specific statement of each step (usually the scenario's
  numbered step) as the STEP.
- DUPLICATE: the same step as another requirement (same actor doing the same
  thing, in other words). belongs_to = that STEP.
- DETAIL: how a step is carried out (mechanism, protocol, algorithm,
  example, data it contains) or a constraint/assumption on it. belongs_to =
  that STEP.
- CONTEXT: goals, background, benefits, definitions; no behaviour.

An expert diagram of a specification has one action per main step: aim for
the number of distinct steps a reader would draw, not one per sentence.
Distinct steps stay distinct: different actors, send vs receive, a decision
vs the action it guards, and every exception and termination case are
separate STEPs. Never drop a behaviour that appears only once.
Return one verdict per requirement ID, copying IDs exactly.
"""


class ConsolidationAgent:
    """Removes restatements and explanations from a chunk-wise extracted list."""

    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(self, requirements: list[Requirement]) -> ConsolidationResult:
        listing = "\n".join(
            f"- {r.id} [{r.type.value}] ({', '.join(r.actors) or '-'}) {r.text}"
            for r in requirements
        )
        user = f"REQUIREMENTS\n============\n{listing}\n\nClassify every requirement."
        return self.llm.complete(system=_SYSTEM_PROMPT, user=user, response_model=ConsolidationResult)


def apply_consolidation(requirements: list[Requirement], result: ConsolidationResult) -> int:
    """Duplicates, details and context become OTHER: kept for traceability, not drawn.

    Returns how many requirements were set aside.
    """
    by_id = {r.id: r for r in requirements}
    steps = {v.id for v in result.verdicts if v.kind.upper() == "STEP"}
    changed = 0
    for verdict in result.verdicts:
        requirement = by_id.get(verdict.id)
        kind = verdict.kind.upper()
        if requirement is None or kind == "STEP" or requirement.type == RequirementType.OTHER:
            continue
        # A duplicate or detail must point at a kept step, or it stays.
        if kind in ("DUPLICATE", "DETAIL") and verdict.belongs_to not in steps:
            continue
        if kind not in ("DUPLICATE", "DETAIL", "CONTEXT"):
            continue
        requirement.type = RequirementType.OTHER
        changed += 1
    return changed
