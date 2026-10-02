from __future__ import annotations

from src.llm.openai_client import OpenAIClient
from src.models.domain import DecompositionPlan, Requirement
from src.pipeline.requirement_consolidation import compact_requirement_lines


class DecompositionAgent:
    """Splits a large requirement set into cohesive sub-activities.

    Each module is later planned, generated, validated and repaired on its
    own, so every downstream prompt stays small no matter how long the
    source specification is. The agent also proposes the control flow
    between modules, which becomes the overview diagram.
    """

    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(
        self,
        requirements: list[Requirement],
        max_requirements_per_module: int,
        document_title: str = "",
    ) -> DecompositionPlan:
        system = f"""
You are a senior software architect who structures large specifications into
UML activity models.

TASK
Partition the atomic requirements into cohesive MODULES (sub-activities such
as use cases, business processes or system phases) and describe the control
flow between them.

PRIVATE VERIFICATION CHECKLIST
Silently:
1. Identify the distinct workflows / use cases / phases of the system.
2. Assign EVERY requirement ID to EXACTLY ONE module.
3. Keep requirements of one workflow together, even across sections.
4. Keep each module at or below {max_requirements_per_module} requirements;
   split a large workflow into consecutive phases if needed.
5. Order modules in the natural execution order of the system.
6. Describe how control passes between modules using transitions.

Do not expose internal reasoning. Return only the structured plan.

RULES
- Module IDs are M1, M2, M3 ... in execution order.
- Module names are short verb phrases ("Register Service", "Negotiate SLA").
- objective: one sentence describing what the module achieves.
- actors: the system's swimlanes, 3 to 8 participants that perform its
  steps (people, organisations or system components), named as in the
  specification's actor lists, with abbreviations written out
  ("SR" -> "Service Registry"). Never data, policies, protocols,
  capabilities, algorithms or groups of requirements. One name per
  participant: no singular/plural or "instance" variants.
- lanes: the actors (copied exactly from actors) active in the module.
- transitions use module IDs plus the pseudo IDs START and END.
- There must be a transition from START and at least one into END.
- Sequential phases: START -> M1 -> M2 -> ... -> END, guard null.
- Conditional flow between modules: one transition per outcome with a
  non-empty guard (e.g. "[resources sufficient]" / "[insufficient]").
- Independent use cases (no ordering between them): START -> each module
  with the guard naming the use case, and each module -> END.
- Specifications often describe the same step twice (an overview or
  product-features section and a detailed use-case section). Put both
  descriptions of a step in the same module, so it is modelled once.
- Do not invent behaviour; only structure what the requirements state.
- Glossary/actor-only requirements still belong to the module where the
  actor or concept is used.
"""
        user = f"""
DOCUMENT
========
{document_title or "Requirements specification"}

ATOMIC REQUIREMENTS ({len(requirements)})
=========================================
{compact_requirement_lines(requirements)}

Produce the DecompositionPlan.
"""
        return self.llm.complete(
            system=system,
            user=user,
            response_model=DecompositionPlan,
        )
