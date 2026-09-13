from __future__ import annotations

from pydantic import BaseModel

from src.llm.openai_client import OpenAIClient
from src.models.domain import Requirement, RequirementMatrix


class RequirementExtractionResponse(BaseModel):
    requirements: list[Requirement]
    matrix: RequirementMatrix


class RequirementAgent:
    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(self, requirement_text: str) -> RequirementExtractionResponse:
        system = """
You are a lead systems requirements engineer and formal specification
specialist with expertise in software requirements and UML activity modeling.

TASK
Decompose the supplied natural-language specification into atomic,
activity-diagram-relevant requirements and initialize an empty traceability
matrix.

PRIVATE REASONING / VERIFICATION CHECKLIST
Silently verify before answering:
1. Read the specification sentence by sentence.
2. Identify every action, condition, decision branch, loop/retry,
   concurrency, exception, data behavior, actor and termination event.
3. Split compound behavior into atomic requirements.
4. Preserve chronological order and causal dependencies.
5. Check that no source behavior was omitted and no behavior was invented.
6. Check IDs are exactly R1, R2, R3... with no gaps.
7. Check every source sentence is represented by one or more requirements.

Do not expose internal reasoning. Return only the structured response.

RULES
- Atomicity: do not combine a condition and action into one undifferentiated
  requirement.
- ZERO HALLUCINATION: never add unstated features, integrations or UI details.
- ZERO OMISSION: include negative paths, retry limits and termination events.
- Use exactly one RequirementType for each requirement.
- Preserve the exact source text in source_sentence.
- Use predecessor requirement IDs in dependencies.
- All list fields must be JSON arrays, never null.
- Initialize matrix with items=[]; downstream stages populate traceability.

TYPE GUIDANCE
ACTION, CONDITION, DECISION, LOOP, CONCURRENCY, EXCEPTION, DATA,
TERMINATION, ACTOR, OTHER.
"""
        user = f"""
REQUIREMENTS DOCUMENT
=====================
{requirement_text}

Extract the complete atomic requirement set.
"""
        return self.llm.complete(
            system=system,
            user=user,
            response_model=RequirementExtractionResponse,
        )
