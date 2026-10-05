from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm.openai_client import OpenAIClient


class LaneAssignment(BaseModel):
    lane: str
    # One of the system's actors, or null when the lane is not an actor
    # (its steps then stay with the actor of the preceding step).
    actor: str | None = None


class LaneMapping(BaseModel):
    assignments: list[LaneAssignment] = Field(default_factory=list)


class LaneMappingAgent:
    """Maps lane names that module diagrams invented onto the system's actors."""

    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(
        self,
        lanes: dict[str, list[str]],
        actors: list[str],
        abbreviations: dict[str, str] | None = None,
    ) -> dict[str, str | None]:
        """``lanes`` maps each lane name to example step labels in that lane."""
        system = """
You are a UML activity diagram expert. Swimlanes must name the ACTORS that
perform the steps (people, organisations or system components).

TASK
Each lane below was produced by a sub-diagram and is not one of the system's
actors. Assign every lane to the ONE actor from the list that performs the
steps shown for it. Use the steps, not just the lane name: "Deploy functional
policies" in a lane "functional policies" is performed by whichever actor
deploys policies.

Abbreviations mean what the specification's glossary says ("PA" may be
"Peering Agent", never guess "Policy Agent" from the letters). Never map a
lane to an actor of a different kind just because one must be chosen (an
organisation is not an end-user).

Answer null only when none of the actors performs those steps.
Copy the lane names and actor names exactly.
"""
        listing = "\n".join(
            f"- {lane}: " + "; ".join(steps[:4]) for lane, steps in lanes.items()
        )
        glossary = "\n".join(f"- {abbr} = {name}" for abbr, name in (abbreviations or {}).items())
        user = f"""
GLOSSARY (abbreviations defined by the specification)
=====================================================
{glossary or "- none"}

ACTORS
======
{chr(10).join(f"- {actor}" for actor in actors)}

LANES (name: example steps)
===========================
{listing}

Return one assignment per lane.
"""
        response = self.llm.complete(system=system, user=user, response_model=LaneMapping)
        allowed = {actor.casefold(): actor for actor in actors}
        result: dict[str, str | None] = {}
        for assignment in response.assignments:
            if assignment.lane not in lanes:
                continue
            actor = (assignment.actor or "").strip()
            result[assignment.lane] = allowed.get(actor.casefold())
        return result


class ActorList(BaseModel):
    actors: list[str] = Field(default_factory=list)


class ActorAgent:
    """Chooses the swimlanes (actors) of a diagram before it is planned."""

    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(self, requirement_text: str) -> list[str]:
        system = """
You are a UML activity diagram expert choosing the SWIMLANES of one diagram.

A swimlane is a participant that performs steps: a person, organisation,
subsystem or component. Choose 2 to 8 of them.
- Use the specification's own actor names, with abbreviations written out
  exactly as the specification defines them ("Peering Agent (PA)": "PA" is
  "Peering Agent"). When the specification declares actors explicitly
  ("Actor: X", an actor list), use those.
  Steps the system performs go in a lane named after the system.
- Every lane must perform at least one step of the described behaviour.
- A capability, function or operation is NOT a lane ("Encryption/Decryption",
  "Matching", "Billing"): its steps go in the lane of whoever performs them.
  Neither are places or groupings ("Local Network", "Multicast Group").
- When participants of the same kind exchange messages (one agent sends,
  another receives), name the ROLES: "Sending Agent", "Receiving Agent",
  and do not also add the generic participant ("Agent").
- When the specification distinguishes local and external (or primary and
  peer) instances of a participant, keep the qualifier of the one whose
  steps are described: "Local Peering Agent".
- Never use data, messages, policies, protocols, capabilities, or vague
  names such as "system" when the specification names the system.
- One name per participant (no singular/plural or "instance" variants).
Return the lanes in the order they first act.
"""
        user = f"SPECIFICATION\n=============\n{requirement_text}\n\nReturn the swimlanes."
        return [a.strip() for a in self.llm.complete(system=system, user=user, response_model=ActorList).actors if a.strip()]
