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

OBJECTIVE
Repair ONLY the supplied defects while preserving all correct behavior.
Return the COMPLETE ActivityDiagram in the required schema.

PRIVATE REASONING / VERIFICATION CHECKLIST
Before producing the result, silently:
1. Locate each supplied defect in the current graph.
2. Determine the smallest graph change that resolves it.
3. Check all incoming/outgoing edges affected by the change.
4. Check requirement traceability before and after the change.
5. Check decisions, loops, exceptions, concurrency and termination.
6. Check that no unrelated behavior was changed.
7. Check exactly one initial node, at least one final node, valid references,
   explicit decision guards, valid edge types and requirement ID lists.
8. Check that the generated graph is suitable for deterministic PlantUML
   compilation.

Do not expose this internal reasoning or chain-of-thought. Return only the
structured RepairResult plus concise change descriptions.

REPAIR RULES
1. Fix ONLY the supplied defects.
2. Prefer the smallest valid modification.
3. Preserve correct nodes and edges.
4. Preserve requirement IDs and valid traceability.
5. Never invent requirements or new business behavior.
6. Never redesign the whole diagram.
7. Do not remove correctly represented behavior.
8. Preserve the exact node ID format N1, N2, ... and edge ID format E1, E2, ...
9. Preserve every existing node lane unless the supplied defect explicitly
   requires changing actor ownership.
10. Exactly one initial node and at least one final node.
11. Every edge must contain type, guard and requirement_ids.
12. Decision outgoing edges must have non-empty guards and distinct guards for
    distinct outcomes.
13. Loops must preserve one entry, one body, and one exit; do not duplicate
    loop bodies or add stops inside ordinary branches.
14. Parallel flows must preserve one fork, one branch per concurrent activity,
    and one join before sequential flow resumes.
15. Do not convert actor, subsystem, or capability descriptions into
    decisions. Keep them in lanes or as actions/notes.
16. If a defect cannot be fixed without unrelated changes, make the minimum
    necessary change and describe it briefly.

SUCCESS CONDITION
The supplied defects should be resolved without introducing new structural
or semantic behavior.
"""
        user = f"""
ORIGINAL REQUIREMENTS
=====================
{requirement_text}

ATOMIC REQUIREMENTS
===================
{[r.model_dump() for r in requirements]}

CURRENT ACTIVITY DIAGRAM
========================
{diagram.model_dump()}

SUPPLIED DEFECTS
================
{[d.model_dump() for d in defects]}

Return the complete repaired ActivityDiagram.
"""
        return self.llm.complete(
            system=system,
            user=user,
            response_model=RepairResult,
        )
