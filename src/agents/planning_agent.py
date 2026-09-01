from __future__ import annotations

from src.llm.openai_client import OpenAIClient
from src.models.domain import (
    ActivityPlan,
    Requirement,
    RequirementMatrix,
)


class PlanningAgent:

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
    ) -> ActivityPlan:

        system = """
You are an expert requirements engineer and UML Activity Diagram planner.

Your job is to convert natural-language requirements into an explicit
Activity Diagram plan BEFORE generation.

The plan will later be compiled into PlantUML.

Never invent business behavior.

Every atomic requirement must be traceable.

You MUST explicitly identify:

1. Sequential actions
2. Decisions
3. Decision guards
4. Loops
5. Loop entry
6. Loop body
7. Loop exit
8. Concurrency
9. Forks
10. Joins
11. Exceptions
12. Termination

IMPORTANT LOOP RULE:

If the requirements describe retry/repetition, do NOT represent it merely
as a back-edge.

Create a LoopPlan.

For every loop specify:

- loop_id
- condition
- entry_node
- body_nodes
- exit_node
- continue_guard
- exit_guard

IMPORTANT DECISION RULE:

For every decision:

- condition
- branch IDs
- guard for each branch

IMPORTANT CONCURRENCY RULE:

For parallel activities:

- fork ID
- branch IDs
- join ID

The resulting plan must be suitable for deterministic compilation into
PlantUML.

All fields are mandatory.

Use [] for empty lists.

Use null only where a guard is genuinely absent.
"""

        user = f"""
ORIGINAL REQUIREMENTS
=====================

{requirement_text}


ATOMIC REQUIREMENTS
===================

{[r.model_dump() for r in requirements]}


EXISTING TRACEABILITY
=====================

{matrix.model_dump()}


Produce the complete ActivityPlan.
"""

        return self.llm.complete(
            system=system,
            user=user,
            response_model=ActivityPlan,
        )