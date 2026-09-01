from __future__ import annotations

from src.llm.openai_client import OpenAIClient
from src.models.domain import ActivityDiagram, Defect, RepairResult, Requirement


class LLMRepairAgent:
    def __init__(self, llm: OpenAIClient) -> None:
        self.llm = llm

    def run(
        self,
        requirement_text: str,
        requirements: list[Requirement],
        diagram: ActivityDiagram,
        defects: list[Defect],
    ) -> RepairResult:
        system = """
        You are an expert UML Activity Diagram repair agent.

        You receive:

        1. Original requirements
        2. Current activity diagram
        3. Specific detected defects

        Your task is to produce a corrected Activity Diagram.

        CRITICAL REPAIR RULES:

        1. Fix ONLY the supplied defects.
        2. Preserve all correct nodes.
        3. Preserve all correct edges.
        4. Preserve requirement IDs.
        5. Do not invent requirements.
        6. Do not remove correctly represented requirements.
        7. Do not change behavior unrelated to the supplied defects.
        8. Do not redesign the entire diagram.
        9. Prefer the smallest valid modification.
        10. If a defect cannot be fixed without changing unrelated behavior,
            preserve the existing behavior and make the minimum necessary change.
        11. Every existing requirement trace must remain present unless the
            corresponding diagram element is explicitly identified as erroneous.
        12. The resulting diagram must contain exactly one initial node.
        13. The resulting diagram must contain at least one final node.
        14. Decision nodes must have explicit guards.
        15. Loops must preserve their intended entry, body, and exit behavior.
        16. Parallel flows must preserve synchronization.

        REPAIR SUCCESS CRITERION:

        After your repair, the supplied defects should be resolved while
        introducing no new structural or semantic behavior.

        Return the COMPLETE ActivityDiagram.
        """
        user = f"""Requirements:
{requirement_text}

Atomic requirements:
{[r.model_dump() for r in requirements]}

Current diagram:
{diagram.model_dump()}

Defects:
{[d.model_dump() for d in defects]}

Return the repaired diagram plus concise change descriptions."""
        return self.llm.complete(
            system=system,
            user=user,
            response_model=RepairResult,
        )
