from src.models.domain import (
    ActivityDiagram,
    ActivityNode,
    CandidateMetrics,
    CandidateRecord,
    Defect,
    NodeType,
    PipelineState,
    Severity,
)
from src.reporting.iterations import ADDED, FLAGGED, highlight, iteration_views

PREFIXES = ("D-COV", "D-STRUCT")


def _candidate(iteration: int, labels: list[str], defects: list[Defect], traces=None) -> CandidateRecord:
    traces = traces or {}
    nodes = [ActivityNode(id="N1", type=NodeType.INITIAL, label="Start")]
    nodes += [
        ActivityNode(id=f"N{i + 2}", type=NodeType.ACTION, label=label, requirement_ids=traces.get(label, []))
        for i, label in enumerate(labels)
    ]
    plantuml = "\n".join(["@startuml", "start", *[f":{label};" for label in labels], "stop", "@enduml"])
    return CandidateRecord(
        iteration=iteration,
        diagram=ActivityDiagram(title="t", nodes=nodes, edges=[]),
        plantuml=plantuml,
        metrics=CandidateMetrics(iteration=iteration),
        defects=defects,
    )


def test_highlight_colours_the_nth_box_with_a_repeated_label():
    candidate = _candidate(0, ["Encrypt", "Send", "Encrypt"], [])
    text = highlight(candidate.plantuml, candidate, {"N4": FLAGGED})
    assert text.splitlines()[2:5] == [":Encrypt;", ":Send;", f"{FLAGGED}:Encrypt;"]


def test_missing_requirement_is_resolved_and_the_new_step_is_green():
    missing = Defect(id="D-COV-R2", category="REQUIREMENT_COVERAGE", severity=Severity.HIGH,
                     description="Requirement R2 is not traced.", requirement_ids=["R2"])
    duplicate = Defect(id="D-GRAN-DUP-N2", category="GRANULARITY", severity=Severity.MEDIUM,
                       description="Step drawn twice.", node_ids=["N2"])
    first = _candidate(0, ["Send"], [missing, duplicate, missing])
    renumbered = duplicate.model_copy(update={"id": "D-GRAN-DUP-N3", "node_ids": ["N3"]})
    second = _candidate(1, ["Notify", "Send"], [renumbered], traces={"Notify": ["R2"]})
    state = PipelineState(
        sample_id="s", requirement_text="", candidates=[second, first],
        repair_history=[{"iteration": 0, "accepted": True, "reason": "FEWER_BLOCKING_DEFECTS",
                         "defect_ids": ["D-COV-R2", "D-GRAN-DUP-N2"], "changes": ["Added Notify."]}],
    )
    views = iteration_views(state, PREFIXES)

    assert [v.iteration for v in views] == [0, 1]
    assert [d.id for d in views[0].defects] == ["D-COV-R2", "D-GRAN-DUP-N2"]
    assert views[0].blocking_ids == {"D-COV-R2"}
    assert f"{FLAGGED}:Send;" in views[0].plantuml
    assert [d.id for d in views[1].resolved] == ["D-COV-R2"]
    assert views[1].added_steps == ["Notify"]
    assert f"{ADDED}:Notify;" in views[1].plantuml
    # The duplicate's node was renumbered (N2 -> N3) but is the same defect.
    assert all(d.category != "GRANULARITY" for d in views[1].resolved)
