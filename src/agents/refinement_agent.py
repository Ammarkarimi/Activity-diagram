from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm.openai_client import OpenAIClient
from src.models.domain import ActivityDiagram, Requirement
from src.utils.json_utils import prompt_json


class RefinementResult(BaseModel):
    # What was changed and why, one line per change (empty if nothing).
    changes: list[str] = Field(default_factory=list)
    diagram: ActivityDiagram


class RefinementAgent:
    """Reworks a first-draft diagram the way an experienced modeller would.

    The planner works requirement by requirement and tends to draw
    alternatives as separate copies of the same steps, and to leave out the
    other actors' steps. This pass looks at the whole draft at once.
    """

    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(
        self,
        requirement_text: str,
        requirements: list[Requirement],
        diagram: ActivityDiagram,
    ) -> RefinementResult:
        system = """
You are an experienced UML modeller reviewing a first-draft activity diagram
against its specification. Return an improved diagram in the same IR.

APPLY THESE CHECKS, IN ORDER
1. Duplicated alternatives: when two or more branches of a decision perform
   the same steps for different variants (save buy instruction / save sell
   instruction -> match -> trade), merge them into ONE path whose decision
   guard names the variants ("buy / sell"). Keep separate only the steps
   that really differ.
2. Shared steps: a step every branch performs before or after a decision
   (encrypt, validate, save, notify, decrypt) appears once, outside it.
3. Actors' steps: every scenario step names who performs it. Each such step
   is an action in that actor's lane; an exchange between two actors is the
   sender's step followed by the receiver's step. Add steps the
   specification states but the draft leaves out.
4. Exceptions: each exception the specification lists is a decision branch
   that ends (final node) or rejoins, as the text says. Merge exception
   branches that end identically into one guard ("no match / exception").
5. Labels: short verb phrases without the actor's name; decisions are short
   questions; guards a few words.

CONSTRAINTS
- No behaviour the specification does not state. Do not draw goals,
  benefits, open issues or examples (requirement types OTHER, ACTOR).
- Keep every lane name that is already one of the specification's actors.
- Every behavioural requirement ID in the draft stays traced to at least one
  node or edge (merge requirement_ids when you merge nodes).
- Exactly one initial node; at least one final node; every node reachable;
  every decision has two or more guarded outgoing edges.
- If the draft already satisfies all checks, return it unchanged with an
  empty change list.
"""
        user = f"""
SPECIFICATION
=============
{requirement_text}

REQUIREMENTS
============
{prompt_json(requirements)}

FIRST-DRAFT DIAGRAM
===================
{prompt_json(diagram)}

Return the list of changes and the improved diagram.
"""
        return self.llm.complete(system=system, user=user, response_model=RefinementResult)
