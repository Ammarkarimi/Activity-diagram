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

    def run(self, lanes: dict[str, list[str]], actors: list[str]) -> dict[str, str | None]:
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

Answer null only when none of the actors performs those steps.
Copy the lane names and actor names exactly.
"""
        listing = "\n".join(
            f"- {lane}: " + "; ".join(steps[:4]) for lane, steps in lanes.items()
        )
        user = f"""
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
- Use the specification's own actor names, with abbreviations written out.
  Steps the system performs go in a lane named after the system.
- When participants of the same kind exchange messages (one agent sends,
  another receives), name the ROLES: "Sending Agent", "Receiving Agent".
- Never use data, messages, policies, protocols, capabilities, or vague
  names such as "system" when the specification names the system.
- One name per participant (no singular/plural or "instance" variants).
Return the lanes in the order they first act.
"""
        user = f"SPECIFICATION\n=============\n{requirement_text}\n\nReturn the swimlanes."
        return [a.strip() for a in self.llm.complete(system=system, user=user, response_model=ActorList).actors if a.strip()]
