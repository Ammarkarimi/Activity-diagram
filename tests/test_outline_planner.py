"""Outline planner: the diagram is planned, repaired and drawn as nested blocks."""
from src.agents.feedback_agent import FeedbackResult
from src.agents.lane_agent import ActorList
from src.agents.outline_agent import OutlineRepairResult
from src.agents.requirement_agent import RequirementExtractionResponse
from src.generation.outline_renderer import OutlineRenderer
from src.models.domain import Defect, NodeType, Severity, Requirement, RequirementMatrix, RequirementType, ReviewResult
from src.models.outline import (
    ActivityOutline,
    NotDrawn,
    OutlineBranch,
    OutlineStep,
    normalize_outline,
    outline_to_diagram,
    requirements_for_outline,
)
from src.pipeline.orchestrator import MultiAgentPipeline
from src.validation.reachability import reachable_node_ids
from src.validation.structural.structural_validator import StructuralValidator


def act(label, lane, *reqs):
    return OutlineStep(kind="action", label=label, lane=lane, requirement_ids=list(reqs))


def sample_outline():
    return ActivityOutline(title="Peering", steps=[
        act("Detect resources", "Web Server", "R1"),
        OutlineStep(kind="decision", label="Overload detected?", lane="Web Server", requirement_ids=["R2"], branches=[
            OutlineBranch(guard="no", steps=[act("Continue delivery", "Web Server", "R2"), OutlineStep(kind="stop")]),
            OutlineBranch(guard="yes", steps=[act("Send initiation request", "Web Server", "R3")]),
        ]),
        act("Receive initiation request", "Mediator", "R3"),
        OutlineStep(kind="loop", label="Sufficient resources?", lane="Mediator", exit_guard="yes",
                    requirement_ids=["R4"], branches=[
                        OutlineBranch(guard="no", steps=[act("Re-negotiate", "Local Peering Agent", "R4")])]),
        OutlineStep(kind="fork", lane="Local Peering Agent", branches=[
            OutlineBranch(steps=[act("Exchange load information", "Local Peering Agent", "R5")]),
            OutlineBranch(steps=[act("Exchange accounting information", "Local Peering Agent", "R5")]),
        ]),
        OutlineStep(kind="decision", label="Encrypt?", requirement_ids=["R6"], branches=[
            OutlineBranch(guard="yes", steps=[act("Encrypt", "Mediator", "R6")]),
            OutlineBranch(guard="no"),
        ]),
    ], not_drawn=[NotDrawn(requirement_id="R7", reason="DETAIL of R3"),
                  NotDrawn(requirement_id="R8", reason="DUPLICATE of R99")])


def test_outline_converts_to_a_valid_block_structured_diagram():
    outline = normalize_outline(sample_outline())
    diagram = outline_to_diagram(outline)
    ids = [s.id for s in outline.steps]
    assert ids[0] == "N2"
    assert {n.id for n in diagram.nodes} == reachable_node_ids(diagram)
    serious = [d for d in StructuralValidator().validate(diagram).defects if d.severity.value in ("HIGH", "CRITICAL")]
    assert serious == []
    # the outline step IDs are the node IDs defects refer to
    assert {s.id for s in outline.steps} <= {n.id for n in diagram.nodes}
    assert sum(n.type == NodeType.FINAL for n in diagram.nodes) == 2


def test_outline_is_drawn_as_nested_blocks():
    text = OutlineRenderer().render(normalize_outline(sample_outline()))
    assert "while (Sufficient resources?) is (no)" in text
    assert "endwhile (yes)" in text
    assert "fork again" in text and "end fork" in text
    assert "detach" not in text and "(A)" not in text
    assert text.rstrip().endswith("stop\n\n@enduml") or "stop" in text.splitlines()[-3]


def test_only_justified_not_drawn_entries_are_set_aside():
    reqs = [Requirement(id=f"R{i}", text="x", type=RequirementType.ACTION) for i in range(1, 9)]
    result, aside = requirements_for_outline(reqs, normalize_outline(sample_outline()))
    assert aside == ["R7"]  # R8 points at a step that is not drawn
    assert {r.id: r.type for r in result}["R7"] == RequirementType.OTHER
    assert reqs[6].type == RequirementType.ACTION  # input untouched


class FakeLLM:
    def __init__(self, review_defect=False):
        self.calls = []
        self.review_defect = review_defect

    def complete(self, *, system, user, response_model):
        self.calls.append(response_model.__name__)
        if response_model is RequirementExtractionResponse:
            return RequirementExtractionResponse(
                requirements=[Requirement(id=f"R{i}", text=f"step {i}", type=RequirementType.ACTION) for i in range(1, 8)],
                matrix=RequirementMatrix(),
            )
        if response_model is ActorList:
            return ActorList(actors=["Web Server", "Mediator", "Local Peering Agent"])
        if response_model is ActivityOutline:
            return sample_outline()
        if response_model is ReviewResult:
            if self.review_defect and self.calls.count("ReviewResult") == 1:
                return ReviewResult(semantic_score=0.5, defects=[Defect(
                    id="SEM-1", category="SEMANTIC", severity=Severity.HIGH, description="Step missing",
                    node_ids=["N2"], edge_ids=[], requirement_ids=[], evidence="", suggested_action="add it")])
            return ReviewResult(semantic_score=1.0)
        if response_model is FeedbackResult:
            return FeedbackResult()
        if response_model is OutlineRepairResult:
            return OutlineRepairResult(outline=sample_outline())
        raise AssertionError(response_model)


def test_outline_pipeline_end_to_end_keeps_initial_and_final_plantuml(tmp_path):
    llm = FakeLLM()
    state = MultiAgentPipeline(llm=llm, planner="outline").run(
        sample_id="peer", requirement_text="A short spec.", max_iterations=1, output_dir=tmp_path, render=False,
    )
    assert "ActivityOutline" in llm.calls and "ActivityPlan" not in llm.calls
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    initial = (run_dir / "initial.puml").read_text(encoding="utf-8")
    assert "while (Sufficient resources?)" in initial
    assert "while (Sufficient resources?)" in state.final_plantuml
    assert state.metrics["before_repair"]["plantuml_file"].endswith("initial.puml")
    assert state.metrics["planner"] == "outline"
    assert state.outline is not None


def test_outline_repair_path_runs(tmp_path):
    llm = FakeLLM(review_defect=True)
    state = MultiAgentPipeline(llm=llm, planner="outline").run(
        sample_id="peer", requirement_text="A short spec.", max_iterations=2, output_dir=tmp_path, render=False,
    )
    assert "OutlineRepairResult" in llm.calls
    assert state.repair_history and state.repair_history[0]["repair_type"] == "OUTLINE"


def test_outline_checks_flag_fork_stops_duplicate_guards_and_repeated_steps():
    from src.models.outline import outline_defects

    outline = normalize_outline(ActivityOutline(title="T", steps=[
        OutlineStep(kind="fork", branches=[
            OutlineBranch(steps=[act("Buy", "UI"), OutlineStep(kind="stop")]),
            OutlineBranch(steps=[act("Sell", "UI")]),
        ]),
        OutlineStep(kind="decision", label="Type?", branches=[
            OutlineBranch(guard="yes", steps=[act("Save", "CTS")]),
            OutlineBranch(guard="Yes", steps=[act("Save", "CTS")]),
        ]),
    ]))
    ids = {d.id.rsplit("-", 1)[0] for d in outline_defects(outline)}
    assert ids == {"D-CON-OUT-STOP", "D-DECISION-OUT-GUARD", "D-GRAN-DUP"}
    assert outline_defects(normalize_outline(sample_outline())) == []
