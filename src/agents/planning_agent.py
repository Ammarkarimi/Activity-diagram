from __future__ import annotations

from src.llm.openai_client import OpenAIClient
from src.utils.json_utils import prompt_json
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
- No omitted behaviour: every ACTION, CONDITION, DECISION, LOOP,
    CONCURRENCY, EXCEPTION, DATA and TERMINATION requirement is traced to at
    least one plan node or edge. Requirements of type OTHER (goals,
    benefits, background, open questions, examples) and ACTOR are context:
    do not draw them; use ACTOR requirements to name the lanes.
- Every node has a unique P1, P2... temp_id.
- Every action, decision, fork, join, object and note has a lane: the
    actor (person, organisation, subsystem or component) that performs it,
    named as the specification names its actors. Use the same spelling
    everywhere; null only for the initial and final nodes. When the same
    kind of actor plays two roles in one exchange, name the roles
    ("Sending Agent", "Receiving Agent").
- Every decision has explicit, non-empty guards.
- Guards should be mutually exclusive and collectively exhaustive.
- Loops must be formal LoopPlan objects, not unexplained back-edges.
- Concurrency must have matching fork_id, branch_ids and join_id.
- Traceability must cover every behavioural requirement (see above).
- No dangling references and no dead-end action nodes.

MODELLING STYLE (how an experienced UML modeller draws a specification)
- One integrated flow, not one flow per use case. When several use cases are
    alternatives of the same activity (buy / sell / cancel; unicast /
    multicast / broadcast), model the shared steps once and branch with ONE
    decision on the variant where they differ ("Instruction type?",
    "Communication type?"); rejoin after the branch when they share the rest.
- Steps that several variants share before or after the branch (validate,
    encrypt, save, decrypt, notify) are drawn once, outside the branch.
- Interaction between actors: the sender's step ("Send query to Central
    Trading System") in the sender's lane, then the receiver's step
    ("Receive query") in the receiver's lane.
- One action per scenario step, in the scenario's order. Labels are short
    verb phrases (at most 8 words), without the actor's name (the lane shows
    it); decisions are short questions; guards are a few words.
- An exception that ends the scenario is a decision branch that leads to a
    final node (after any reporting step it names).
- Do not draw goals, benefits, quality attributes, background text, open
    issues or examples.

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
{prompt_json(requirements)}

EXISTING TRACEABILITY
=====================
{prompt_json(matrix)}

Produce the complete ActivityPlan.
"""
        return self.llm.complete(
            system=system,
            user=user,
            response_model=ActivityPlan,
        )
