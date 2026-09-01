from __future__ import annotations

from pydantic import BaseModel

from src.llm.openai_client import (
    OpenAIClient,
)
from src.models.domain import (
    Requirement,
    RequirementMatrix,
)


class RequirementExtractionResponse(
    BaseModel
):

    requirements: list[Requirement]

    matrix: RequirementMatrix


class RequirementAgent:

    def __init__(
        self,
        llm: OpenAIClient,
    ) -> None:

        self.llm = llm

    def run(
        self,
        requirement_text: str,
    ) -> RequirementExtractionResponse:

        system = """
You are an expert requirements engineer.

Extract atomic, activity-diagram-relevant
requirements from the supplied document.

Do not invent behavior.

Every requirement must have:

id
text
type
actors
actions
conditions
exceptions
dependencies
source_sentence

Use [] when a list is empty.

Requirement IDs must be stable:

R1, R2, R3, ...

Identify:

- actions
- decisions
- conditions
- loops
- concurrency
- exceptions
- data
- termination
- actors

Also produce an initial RequirementMatrix.

The matrix may contain empty mappings because
the actual mapping is established after
diagram generation.
"""

        user = f"""
REQUIREMENTS DOCUMENT
=====================

{requirement_text}

Extract the requirements.
"""

        return self.llm.complete(
            system=system,
            user=user,
            response_model=(
                RequirementExtractionResponse
            ),
        )