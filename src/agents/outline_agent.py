from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm.openai_client import OpenAIClient
from src.models.domain import Defect, Requirement, modelled_requirements
from src.models.outline import ActivityOutline
from src.utils.json_utils import prompt_json

_STYLE = """
HOW AN EXPERT DRAWS THE DIAGRAM
- ONE diagram, one flow from start to stop, written as nested blocks:
  steps in order, decisions (if/else), loops (while) and forks. Never jump
  back to an earlier part of the flow except through a loop.
- Model the behaviour at the level of the specification's main steps: one
  action per scenario step or per distinct thing an actor does. What a
  component DOES, as its description states it ("a Web server replicates
  content on-demand from the origin server", "ensures delivery of content to
  end-users", "load information is measured and reported"), is a step of
  that component, drawn where it happens in the flow, even when it is only
  stated in a component description. Details of HOW a step is done (named
  algorithms, protocols, levels, examples, data formats), restatements of a
  step already drawn, goals, benefits and background are NOT steps: trace
  them to the step they belong to, or list them in not_drawn.
- Interactions with a repository or registry are a request and an answer:
  "Request policy information from Policy Repository" (requester's lane),
  "Send policy information to Mediator" (repository's lane).
- Every step is in the lane of the actor that performs it (the subject of
  the sentence). An interaction between actors is the sender's step in the
  sender's lane ("Send initiation request to Mediator") followed by the
  receiver's step in the receiver's lane ("Receive initiation request").
- Action labels: a short verb phrase with its object, at most 8 words, no
  actor name (the lane shows it). Decisions and loops: a short question
  ("Sufficient resources acquired?"). Guards: a few words ("yes", "no",
  "unicast", "rejected").
- Alternatives (request types, communication types, outcomes) are ONE
  decision with one branch per alternative; steps the alternatives share
  are drawn once, before or after the decision. Draw every step once: if a
  step would appear in two branches, it belongs before or after them.
- Each decision has one branch per distinct outcome; never two branches
  with the same guard, never a decision whose branches all do nothing.
- An optional step ("encrypt if requested") is a decision whose other
  branch is empty: the flow continues after the decision either way.
- A negative outcome that ends the behaviour (request rejected, failure) is
  a branch with its reporting step(s) followed by a stop. A branch that
  just continues has no stop.
- "Repeat / re-negotiate / retry until X" is a loop: label "X?" style
  question, the body runs while X does not hold (body guard "no",
  exit_guard "yes"). Put the steps that are repeated in the body.
- Activities that go on together without an order between them (exchanging
  different kinds of information, sending a result to two receivers) are a
  fork with one branch per activity.
- Nested conditions follow the specification's wording: "if the
  termination condition holds the arrangement is disbanded or re-arranged,
  except in exceptional circumstances" is: termination condition? -> yes:
  exceptional circumstances? -> no: re-arrange or disband?
- Use cases that are phases of one lifecycle follow each other; use cases
  that are alternatives are branches of one decision.
- Draw an exception only where the specification says what happens in it.
- A list of conditions that all lead to the same outcome ("the arrangement
  is disbanded when a, b, c or d") is ONE decision ("Termination condition
  holds?"), not one decision per condition.
- The normal operation the specification describes before its use cases,
  when it leads into them (the system receiving and serving requests and
  measuring its load before an overload triggers a use case), is part of the
  flow, drawn before the use case it triggers.
- Never invent steps, actors or conditions the specification does not state.

LANES
- Use only the given swimlanes (copy the names exactly). A step done by the
  system goes in the system's lane.

TRACEABILITY
- Every requirement listed under BEHAVIOURAL REQUIREMENTS is either in the
  requirement_ids of the step(s) that draw it, or in not_drawn with reason
  "DUPLICATE of Rn", "DETAIL of Rn" or "CONTEXT". A behaviour that occurs
  only once in the specification is never not_drawn.
- Decisions and loops carry the IDs of the conditions they test.

STRUCTURE OF THE ANSWER
Fill the analysis fields first, then draw the steps accordingly:
- use_cases: every use case / scenario / phase in the specification, in
  order, with its relation to the previous one: "first", "follows" (a later
  phase of the same lifecycle) or "alternative" (another request type or
  variant the user or system may choose instead: buy / sell / cancel /
  query, unicast / multicast). Alternatives become branches of ONE decision
  ("Instruction type?"); never draw them one after another, never as a fork.
- shared_steps: steps that several alternatives have in common (compose,
  encrypt, receive, decrypt, validate, save). Each is drawn ONCE, before
  the decision on the alternative or after its branches rejoin.
- loops: every "repeat / re-evaluate / re-negotiate / retry until"
  statement, written as "<question>? : <repeated steps>"; each is drawn as a
  loop step (kind "loop") with that question as its label, never an if.
- parallel_activities: groups of activities that happen at the same time;
  each group is drawn as a fork step. A request passed from one actor to the next and then
  processed is a SEQUENCE, never a fork.
- steps: the top-level sequence. Each step has kind:
  action  (label, lane, requirement_ids)
  decision (label = question, lane, branches = one per outcome with guard
            and steps; an empty steps list continues after the decision)
  loop    (label = question, lane, branches = exactly one: the body, its
            guard is the guard under which the body runs; exit_guard)
  fork    (branches = one per parallel path, guard empty)
  stop    (ends this path; label empty or the outcome)
- Leave id empty; IDs are assigned afterwards.
"""


class OutlineAgent:
    """Plans the whole diagram as a block-structured outline."""

    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(
        self,
        requirement_text: str,
        requirements: list[Requirement],
        actors: list[str],
        feedback: str = "",
    ) -> ActivityOutline:
        system = f"""
You are an experienced UML modeller who draws activity diagrams of
requirement specifications by hand, the way they appear in textbooks and
expert reference solutions.

TASK
Write the activity diagram of the specification as an ActivityOutline.
{_STYLE}
"""
        user = f"""
SPECIFICATION
=============
{requirement_text}

SWIMLANES
=========
{chr(10).join(f"- {actor}" for actor in actors) or "- choose the actors the specification names"}

BEHAVIOURAL REQUIREMENTS
========================
{_requirement_listing(modelled_requirements(requirements))}

{("NOTE" + chr(10) + "====" + chr(10) + feedback + chr(10)) if feedback else ""}
Return the complete ActivityOutline.
"""
        return self.llm.complete(system=system, user=user, response_model=ActivityOutline)


class OutlineRepairResult(BaseModel):
    outline: ActivityOutline
    changes: list[str] = Field(default_factory=list)


class OutlineRepairAgent:
    """Repairs defects by editing the outline, so the result stays block-structured."""

    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(
        self,
        requirement_text: str,
        requirements: list[Requirement],
        outline: ActivityOutline,
        defects: list[Defect],
        actors: list[str],
    ) -> OutlineRepairResult:
        system = f"""
You are an experienced UML modeller repairing an activity diagram written as
a block-structured ActivityOutline. Step IDs (N2, N3, ...) are the node IDs
the defects refer to.

TASK
Fix ONLY the supplied defects with the smallest change to the outline, and
return the COMPLETE revised outline plus one line per change.
- Keep every correct step, its lane and its requirement_ids.
- A defect saying a requirement is not traced: add its ID to the step that
  draws it, add the missing step where the specification places it, or,
  when it only restates or explains a drawn step, list it in not_drawn.
- A defect about a dead end, an unreachable step or a missing stop: fix the
  block (a branch either continues after its decision or ends in a stop).
- Never add behaviour the specification does not state; never redesign.
- Keep the same style rules as the original outline:
{_STYLE}
"""
        user = f"""
SPECIFICATION
=============
{requirement_text}

SWIMLANES
=========
{chr(10).join(f"- {actor}" for actor in actors) or "- keep the outline's lanes"}

BEHAVIOURAL REQUIREMENTS
========================
{_requirement_listing(modelled_requirements(requirements))}

CURRENT OUTLINE
===============
{prompt_json(outline)}

DEFECTS TO FIX
==============
{prompt_json(defects)}

Return the complete revised outline and the list of changes.
"""
        return self.llm.complete(system=system, user=user, response_model=OutlineRepairResult)


def _requirement_listing(requirements: list[Requirement]) -> str:
    return "\n".join(
        f"- {r.id} [{r.type.value}] ({', '.join(r.actors) or '-'}) {r.text}" for r in requirements
    ) or "- none"
