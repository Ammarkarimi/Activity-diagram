"""Fixes for issues marked in real runs (outputs/*_marked.png).

- disconnected nodes: branches that start with a lane change or are only a
  stop lost their incoming arrow in swimlane diagrams;
- partial diagrams: unreachable nodes were never drawn but counted as covered,
  and the repair loop stopped when the deterministic repair could not fix them;
- decisions hidden in actions ("Encrypt message" with three guarded exits);
- lanes: "Local PA" mapped to "Policy Agent" although the spec defines
  "Peering Agent (PA)".
"""
import pytest

from src.generation.ir_sanitizer import IRSanitizer
from src.generation.plantuml_generator import PlantUMLGenerator
from src.models.domain import (
    ActivityDiagram,
    ActivityEdge,
    ActivityNode,
    Defect,
    NodeType,
    Requirement,
    RepairResult,
    Severity,
)
from src.pipeline.composer import _base_name
from src.pipeline.lanes import match_actor, spec_abbreviations
from src.pipeline.orchestrator import MultiAgentPipeline, blocking_defects
from src.pipeline.repair_router import RepairRouter
from src.validation.requirements.coverage_validator import CoverageValidator
from src.validation.structural.structural_validator import StructuralValidator
from src.validation.syntax.plantuml_validator import PlantUMLSyntaxValidator

KINDS = {"i": NodeType.INITIAL, "a": NodeType.ACTION, "d": NodeType.DECISION, "f": NodeType.FINAL}


def _diagram(nodes, edges):
    return ActivityDiagram(
        title="T",
        nodes=[
            ActivityNode(id=i, type=KINDS[k], label=label, lane=lane, requirement_ids=list(reqs))
            for i, k, label, lane, reqs in nodes
        ],
        edges=[ActivityEdge(id=f"E{n}", source=s, target=t, guard=g) for n, (s, t, g) in enumerate(edges, 1)],
    )


# ------------------------------------------------------------------ drawing

def test_every_branch_starts_in_the_decision_lane():
    # A stop-only "else" after a branch that moved to lane B used to float in B.
    d = _diagram(
        [("N1", "i", "Start", "A", []), ("N2", "a", "a", "A", []), ("N3", "d", "q?", "A", []),
         ("N4", "a", "b", "B", []), ("N5", "f", "End", None, [])],
        [("N1", "N2", None), ("N2", "N3", None), ("N3", "N4", "x"), ("N3", "N5", "y"), ("N4", "N5", None)],
    )
    lines = PlantUMLGenerator().render(d).splitlines()
    else_at = lines.index("else (y)")
    assert lines[else_at + 1] == "|A|"
    assert lines[else_at + 2] == "stop"


def test_multiway_decision_with_lanes_is_nested_not_elseif():
    d = _diagram(
        [("N1", "i", "Start", "A", []), ("N2", "d", "Type?", "A", []), ("N3", "a", "Buy", "A", []),
         ("N4", "a", "Sell", "A", []), ("N5", "a", "Query", "B", []), ("N6", "f", "End", None, [])],
        [("N1", "N2", None), ("N2", "N3", "buy"), ("N2", "N4", "sell"), ("N2", "N5", "query"),
         ("N3", "N6", None), ("N4", "N6", None), ("N5", "N6", None)],
    )
    text = PlantUMLGenerator().render(d)
    assert "elseif" not in text
    assert text.count("if (Type?)") == 2 and text.count("endif") == 2
    assert "skinparam swimlaneWrapTitleWidth" in text
    for label in ("Buy", "Sell", "Query"):
        assert text.count(f":{label};") == 1


def test_single_lane_diagram_keeps_elseif():
    d = _diagram(
        [("N1", "i", "Start", None, []), ("N2", "d", "Type?", None, []), ("N3", "a", "Buy", None, []),
         ("N4", "a", "Sell", None, []), ("N5", "a", "Query", None, []), ("N6", "f", "End", None, [])],
        [("N1", "N2", None), ("N2", "N3", "buy"), ("N2", "N4", "sell"), ("N2", "N5", "query"),
         ("N3", "N6", None), ("N4", "N6", None), ("N5", "N6", None)],
    )
    text = PlantUMLGenerator().render(d)
    assert "elseif (Type?)" in text and "skinparam" not in text


def test_builtin_check_accepts_link_shapes():
    text = "@startuml\nstart\n:From Part 1<\n:a;\n:Continue in Part 2>\ndetach\n@enduml"
    assert PlantUMLSyntaxValidator._regex_validate(text)[0]
    assert not PlantUMLSyntaxValidator._regex_validate("@startuml\nstart\n:a\nstop\n@enduml")[0]


# ------------------------------------------------------------ validation

def _branching_action():
    return _diagram(
        [("N1", "i", "Start", "A", []), ("N2", "a", "Encrypt message", "A", ["R1"]),
         ("N3", "a", "Send unicast", "A", []), ("N4", "a", "Send multicast", "A", []),
         ("N5", "f", "End", None, [])],
        [("N1", "N2", None), ("N2", "N3", "unicast"), ("N2", "N4", "multicast"),
         ("N3", "N5", None), ("N4", "N5", None)],
    )


def test_validator_flags_action_with_several_exits():
    defects = StructuralValidator().validate(_branching_action()).defects
    assert any(d.id == "D-FLOW-002-N2" for d in defects)


def test_sanitizer_turns_branching_action_into_action_plus_decision():
    d = IRSanitizer.sanitize(_branching_action())
    decision = next(n for n in d.nodes if n.type == NodeType.DECISION)
    assert decision.label == "Unicast or multicast?" and decision.lane == "A"
    assert [e.target for e in d.edges if e.source == "N2"] == [decision.id]
    assert {e.guard for e in d.edges if e.source == decision.id} == {"unicast", "multicast"}
    assert not any(x.id.startswith("D-FLOW-002") for x in StructuralValidator().validate(d).defects)


def test_unreachable_nodes_do_not_count_as_coverage():
    d = _diagram(
        [("N1", "i", "Start", None, []), ("N2", "a", "Shown", None, ["R1"]), ("N3", "f", "End", None, []),
         ("N4", "a", "Implement query", None, ["R2"])],
        [("N1", "N2", None), ("N2", "N3", None), ("N4", "N3", None)],
    )
    reqs = [Requirement(id="R1", text="shown"), Requirement(id="R2", text="query")]
    result = CoverageValidator().validate(d, reqs)
    assert [x.id for x in result.defects] == ["D-COV-R2"]
    assert "unreachable" in result.defects[0].description
    structural = StructuralValidator().validate(d).defects
    unreachable = next(x for x in structural if x.id == "D-STRUCT-003")
    assert unreachable.severity == Severity.CRITICAL
    assert {x.id for x in blocking_defects(result.defects + structural)} >= {"D-COV-R2", "D-STRUCT-003"}


# ---------------------------------------------------------------- repair

def _defect(id_, category):
    return Defect(id=id_, category=category, severity=Severity.HIGH, description=id_)


def test_router_hands_unfixable_structural_defects_to_the_llm():
    class _LLMRepair:
        def run(self, **kwargs):
            return RepairResult(changed=True, repair_type="LLM", changes=["connected"], diagram=kwargs["diagram"])

    router = RepairRouter.__new__(RepairRouter)
    RepairRouter.__init__(router, llm=None)
    router.llm_repair = _LLMRepair()
    d = _diagram([("N1", "i", "Start", None, []), ("N2", "f", "End", None, []), ("N3", "a", "x", None, [])],
                 [("N1", "N2", None)])
    result = router.repair("spec", [], d, [_defect("D-STRUCT-003", "STRUCTURAL")])
    assert result.changed and router.last_used_llm


def test_repair_that_removes_a_blocking_defect_is_accepted_despite_new_findings():
    accepted, reason = MultiAgentPipeline._should_accept_repair(
        original_quality=0.80, repaired_quality=0.81,
        original_defect_count=3, repaired_defect_count=4,
        defects_fixed=1, defects_introduced=2,
        original_blocking={"D-STRUCT-003"}, repaired_blocking=set(),
    )
    assert accepted and reason == "FEWER_BLOCKING_DEFECTS"


def test_candidate_without_blocking_defects_wins():
    from src.models.domain import CandidateMetrics, CandidateRecord

    def record(quality, defects):
        return CandidateRecord(
            iteration=0, diagram=_branching_action(), plantuml="",
            metrics=CandidateMetrics(iteration=0, quality_score=quality, defect_count=len(defects)),
            defects=defects,
        )

    clean = record(0.70, [_defect("SEM-1", "SEMANTIC")])
    broken = record(0.90, [_defect("D-STRUCT-003", "STRUCTURAL")])
    assert MultiAgentPipeline._is_better_candidate(clean, broken)
    assert not MultiAgentPipeline._is_better_candidate(broken, clean)


# ----------------------------------------------------------------- lanes

SPEC = "The mediator passes requirements to the local Peering Agent (PA). The Service Registry (SR) stores them."
ACTORS = ["Mediator", "Service Registry", "Peering Agent", "Policy Agent"]


@pytest.mark.parametrize("lane, actor", [("Local PA", "Peering Agent"), ("PAs", "Peering Agent"), ("SR", "Service Registry")])
def test_lanes_use_the_specs_own_abbreviations(lane, actor):
    assert spec_abbreviations(SPEC) == {"PA": "Peering Agent", "SR": "Service Registry"}
    assert match_actor(lane, ACTORS, spec_abbreviations(SPEC)) == actor


def test_outcome_decision_hides_module_split():
    assert _base_name("Trigger Peering (part 2/2)") == "Trigger Peering"
    assert _base_name("Discover Peers") == "Discover Peers"


def test_unguarded_exits_of_an_action_become_a_fork_not_a_guardless_decision():
    d = _diagram(
        [("N1", "i", "Start", "A", []), ("N2", "a", "Send multicast", "A", []),
         ("N3", "a", "Receive", "B", []), ("N4", "a", "Notify", "A", []), ("N5", "f", "End", None, [])],
        [("N1", "N2", None), ("N2", "N3", None), ("N2", "N4", None), ("N3", "N5", None), ("N4", "N5", None)],
    )
    s = IRSanitizer.sanitize(d)
    fork = next(n for n in s.nodes if n.id == "N2_F")
    assert fork.type == NodeType.FORK
    assert not any(x.id.startswith("D-DECISION-002") for x in StructuralValidator().validate(s).defects)


def test_mixed_batch_sends_rule_fixable_defects_to_rules_and_the_rest_to_the_llm():
    seen = {}

    class _LLMRepair:
        def run(self, **kwargs):
            seen["defects"] = [x.id for x in kwargs["defects"]]
            seen["diagram"] = kwargs["diagram"]
            return RepairResult(changed=True, repair_type="LLM", changes=["named guards"], diagram=kwargs["diagram"])

    router = RepairRouter(llm=None)
    router.llm_repair = _LLMRepair()
    defects = [_defect("D-FLOW-002-N2", "STRUCTURAL"), _defect("D-DECISION-003-N9", "DECISION")]
    result = router.repair("spec", [], _branching_action(), defects)
    assert seen["defects"] == ["D-DECISION-003-N9"]
    # The LLM starts from the rule-repaired diagram, and both changes are kept.
    assert any(n.type == NodeType.DECISION for n in seen["diagram"].nodes)
    assert result.changes[0].startswith("Inserted decision") and result.changes[-1] == "named guards"
    assert router.last_used_llm


def test_actor_list_drops_media_groupings_and_generic_roles():
    from src.pipeline.lanes import canonical_actors

    names = ["Sending Agent", "Receiving Agent", "Multicast Group", "Local Network", "Encryption/Decryption", "Agent"]
    assert canonical_actors(names) == ["Sending Agent", "Receiving Agent"]
    kept = ["Central Trading System", "Trading Information Release", "Service Registry", "Policy Repository"]
    assert canonical_actors(kept) == kept


# ------------------------------------------------------------------ peering feedback

def test_one_unstructurable_loop_keeps_the_other_loops_as_blocks():
    # Negotiation loop (N3..N6) plus a re-arrange edge back into the middle
    # of the flow from a nested branch: only the latter becomes a jump.
    d = _diagram(
        [("N1", "i", "Start", None, []), ("N2", "a", "Pass", None, []), ("N3", "a", "Negotiate", None, []),
         ("N4", "d", "Enough?", None, []), ("N5", "a", "Re-evaluate", None, []), ("N6", "a", "Establish", None, []),
         ("N7", "d", "Terminate?", None, []), ("N8", "d", "Exceptional?", None, []),
         ("N9", "a", "Re-arrange", None, []), ("N10", "a", "Disband", None, []), ("N11", "f", "End", None, [])],
        [("N1", "N2", None), ("N2", "N3", None), ("N3", "N4", None), ("N4", "N5", "no"), ("N5", "N3", None),
         ("N4", "N6", "yes"), ("N6", "N7", None), ("N7", "N10", "yes"), ("N7", "N8", "no"),
         ("N8", "N10", "yes"), ("N8", "N9", "no"), ("N9", "N6", None), ("N10", "N11", None)],
    )
    text = PlantUMLGenerator().render(d)
    assert "repeat while (Enough?)" in text
    assert "detach" in text
    assert PlantUMLGenerator._blocks_balanced(text)


def test_actor_qualifier_follows_the_specification():
    from src.pipeline.lanes import spec_qualified_actors

    text = "The mediator passes them to the local PA. The local PA negotiates. It talks to external PAs."
    abbreviations = {"PA": "Peering Agent"}
    assert spec_qualified_actors(["Mediator", "External Peering Agent"], text, abbreviations) == [
        "Mediator", "Local Peering Agent"
    ]
    assert spec_qualified_actors(["Local Peering Agent"], text, abbreviations) == ["Local Peering Agent"]


def test_long_single_mode_documents_are_extracted_per_chunk():
    # One extraction call over the whole peering spec stopped after 14 requirements.
    from pathlib import Path

    from src.document.chunker import DocumentChunker, estimate_tokens

    text = Path("data/pure/peering.txt").read_text(encoding="utf-8")
    limit = MultiAgentPipeline.CHUNKED_EXTRACTION_TOKENS
    assert estimate_tokens(text) > limit
    assert len(DocumentChunker(max_tokens=limit).chunk(text)) > 1


def test_consolidation_sets_aside_restatements_but_keeps_unanchored_ones():
    from src.agents.consolidation_agent import ConsolidationResult, RequirementVerdict, apply_consolidation

    reqs = [Requirement(id=f"R{i}", text=f"step {i}", type="ACTION") for i in range(1, 5)]
    verdicts = ConsolidationResult(verdicts=[
        RequirementVerdict(id="R1", kind="STEP"),
        RequirementVerdict(id="R2", kind="DUPLICATE", belongs_to="R1"),
        RequirementVerdict(id="R3", kind="DETAIL", belongs_to="R9"),  # no such step: kept
        RequirementVerdict(id="R4", kind="CONTEXT"),
    ])
    assert apply_consolidation(reqs, verdicts) == 2
    assert [r.type.value for r in reqs] == ["ACTION", "OTHER", "ACTION", "OTHER"]
