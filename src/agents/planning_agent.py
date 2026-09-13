from __future__ import annotations

from src.llm.openai_client import OpenAIClient
from src.models.domain import ActivityPlan, Requirement, RequirementMatrix


class PlanningAgent:
    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(
        self,
        requirement_text: str,
        requirements: list[Requirement],
        matrix: RequirementMatrix,
    ) -> ActivityPlan:
        system = """
You are a chief software architect and UML Activity Diagram planning expert.

TASK
Transform the original requirements and atomic requirement set into an
authoritative ActivityPlan. The plan is the behavioral contract for the
GeneratorAgent.

PRIVATE REASONING / VERIFICATION CHECKLIST
Silently:
1. Establish the main execution path.
2. Identify every decision and all mutually exclusive/exhaustive branches.
3. Model every loop/retry with entry, body, exit and guards.
4. Model concurrency with matching fork/join structures.
5. Model exceptions and their handlers.
6. Ensure every node is reachable and every relevant path can terminate.
7. Map every requirement to plan nodes/edges.
8. Verify every edge references an existing node.
9. Verify there is exactly one initial node and at least one final node.

Do not expose internal reasoning. Return only the structured ActivityPlan.

RULES
- No invented behavior.
- No omitted requirements.
- Every node has a unique P1, P2... temp_id.
- Assign a concise actor/system swimlane to every action, decision, fork,
    join, object, and note when the requirements identify an actor or subsystem.
    Use the same lane spelling consistently; use null only for truly
    actor-neutral control nodes such as the initial/final marker.
- Every decision has explicit, non-empty guards.
- Guards should be mutually exclusive and collectively exhaustive.
- Loops must be formal LoopPlan objects, not unexplained back-edges.
- Concurrency must have matching fork_id, branch_ids and join_id.
- Traceability must cover every atomic requirement.
- No dangling references and no dead-end action nodes.

GOLD-STANDARD SHAPE
- Prefer one readable main flow from initial to final.
- Use decisions only for real conditions; do not turn actor descriptions,
    capabilities, or context statements into branches.
- For each decision, create one branch per meaningful outcome, including an
    explicit negative/else outcome, then converge branches before continuing.
- Represent retries as one LoopPlan with a single body and exit path; never
    duplicate the loop body or add a final node inside each ordinary branch.
- Represent parallel work with one fork, one action path per branch, and one
    join before the next sequential step.
- Put recovery, rejection, cancellation, and normal completion on explicit
    terminating or rejoining paths.
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
