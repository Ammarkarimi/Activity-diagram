from __future__ import annotations

from src.llm.openai_client import OpenAIClient
from src.models.domain import ActivityDiagram, ActivityPlan, Requirement


class GeneratorAgent:
    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(
        self,
        requirement_text: str,
        requirements: list[Requirement],
        plan: ActivityPlan,
    ) -> ActivityDiagram:
        system = """
You are an expert UML Activity Diagram graph generator.

TASK
Convert the supplied ActivityPlan into a canonical ActivityDiagram IR.
The ActivityPlan is authoritative.

PRIVATE REASONING / VERIFICATION CHECKLIST
Silently:
1. Convert each plan node to exactly one diagram node where appropriate.
2. Convert each plan edge without changing its behavior.
3. Preserve decisions, guards, loops, exceptions, concurrency and termination.
4. Preserve every requirement trace.
5. Verify all edge endpoints exist.
6. Verify exactly one initial node and at least one final node.
7. Verify every decision has guards on all outgoing edges.
8. Verify forks have at least two branches and joins synchronize them.
9. Verify IDs follow N1/N2/... and E1/E2/....
10. Verify the result is compilable by the deterministic PlantUML generator.

Do not expose internal reasoning. Return only the structured ActivityDiagram.

RULES
- Do not invent new workflow behavior.
- Do not remove behavior from the plan.
- Use node types: initial, final, action, decision, merge, fork, join, object, note.
- Copy each plan node's lane into the diagram node unchanged. Lanes represent
    actors, services, repositories, or subsystems and must use stable concise
    names such as "Web Server", "Mediator", or "Policy Repository".
- Use edge types: control or object.
- Every edge includes type, guard and requirement_ids.
- Use null for an unguarded edge; use non-empty strings for decision guards.
- Use [] for empty requirement mappings.
- Preserve all requirement IDs from the plan.

GOLD-STANDARD SHAPE
- Produce one connected activity flow with exactly one initial node and at
    least one final node.
- Keep actor/system responsibilities in lanes instead of inventing decisions
    such as "start processing" or repeating the same condition for each actor.
- A decision must have one outgoing edge per real outcome, with distinct
    non-empty guards. Do not create nested decisions that restate the same
    condition.
- A loop must have one body and one exit; do not emit duplicate loop bodies,
    duplicate repeat constructs, or stop statements inside ordinary branches.
- A fork must have at least two parallel branches and a join before the main
    flow continues.
- Use final nodes only for genuine workflow termination: success, rejection,
    cancellation, failure, or an explicitly terminal requirement.
"""
        user = f"""
ORIGINAL REQUIREMENTS
=====================
{requirement_text}

ATOMIC REQUIREMENTS
===================
{[r.model_dump() for r in requirements]}

AUTHORITATIVE ACTIVITY PLAN
===========================
{plan.model_dump()}

Generate the complete ActivityDiagram.
"""
        return self.llm.complete(
            system=system,
            user=user,
            response_model=ActivityDiagram,
        )
