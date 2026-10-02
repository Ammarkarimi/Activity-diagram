"""Structure of the generated PlantUML: loops, merge points and composition.

Each diagram below reproduces a shape seen in a real hierarchical run
(outputs/peering_*) that used to produce duplicated or missing output.
"""
from collections import Counter

from src.generation.plantuml_generator import PlantUMLGenerator
from src.models.domain import (
    ActivityDiagram,
    ActivityEdge,
    ActivityNode,
    DecompositionPlan,
    ModuleTransition,
    ActivityModule,
    NodeType,
)
from src.pipeline.composer import build_overview_diagram, compose_full_diagram
from src.validation.structural.structural_validator import StructuralValidator


def _diagram(nodes, edges, title="T"):
    kinds = {
        "i": NodeType.INITIAL,
        "a": NodeType.ACTION,
        "d": NodeType.DECISION,
        "f": NodeType.FINAL,
        "k": NodeType.FORK,
        "j": NodeType.JOIN,
    }
    return ActivityDiagram(
        title=title,
        nodes=[ActivityNode(id=i, type=kinds[k], label=label) for i, k, label in nodes],
        edges=[
            ActivityEdge(id=f"E{n}", source=s, target=t, guard=g)
            for n, (s, t, g) in enumerate(edges, start=1)
        ],
    )


def _actions(text):
    return [line for line in text.splitlines() if line.startswith((":", "backward:"))]


def _assert_each_action_once(text, diagram):
    actions = _actions(text)
    expected = sum(1 for n in diagram.nodes if n.type == NodeType.ACTION)
    duplicates = [a for a, c in Counter(actions).items() if c > 1]
    assert not duplicates, duplicates
    assert len(actions) == expected
    assert PlantUMLGenerator._blocks_balanced(text)


def test_branches_merge_before_repeat_loop_closes():
    # Two branches rejoin at N7 inside a loop that goes back to N2 (module M5).
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N2", "a", "Receive"),
            ("N4", "d", "Existing?"),
            ("N5", "a", "Reuse"),
            ("N6", "a", "Negotiate"),
            ("N7", "a", "Measure load"),
            ("N12", "d", "Enough resources?"),
            ("N13", "a", "Establish"),
            ("N14", "a", "Re-evaluate"),
            ("N15", "f", "Done"),
        ],
        [
            ("N1", "N2", None),
            ("N2", "N4", None),
            ("N4", "N5", "yes"),
            ("N4", "N6", "no"),
            ("N5", "N7", None),
            ("N6", "N7", None),
            ("N7", "N12", None),
            ("N12", "N13", "yes"),
            ("N12", "N14", "no"),
            ("N13", "N15", None),
            ("N14", "N2", None),
        ],
    )
    text = PlantUMLGenerator().render(d)
    _assert_each_action_once(text, d)
    lines = text.splitlines()
    assert lines.index("repeat") < lines.index(":Receive;")
    assert "backward:Re-evaluate;" in lines
    assert "repeat while (Enough resources?) is (no) not (yes)" in lines


def test_decision_at_loop_head_becomes_while():
    # N4 -> N11 -> N4 retry loop (module M6).
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N3", "a", "Advertise"),
            ("N4", "d", "Established?"),
            ("N5", "a", "Exchange"),
            ("N11", "a", "Re-negotiate"),
            ("N13", "f", "Done"),
        ],
        [
            ("N1", "N3", None),
            ("N3", "N4", None),
            ("N4", "N5", "yes"),
            ("N4", "N11", "no"),
            ("N11", "N4", None),
            ("N5", "N13", None),
        ],
    )
    text = PlantUMLGenerator().render(d)
    _assert_each_action_once(text, d)
    lines = text.splitlines()
    start = lines.index("while (Established?) is (no)")
    assert lines[start + 1] == ":Re-negotiate;"
    assert lines[start + 2] == "endwhile (yes)"
    assert lines[start + 3] == ":Exchange;"


def test_nested_decisions_sharing_a_merge_point_emit_it_once():
    # N6 and the nested N10 both rejoin at N12 (module M6).
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N6", "d", "Enough?"),
            ("N7", "a", "Deploy"),
            ("N10", "d", "Disband?"),
            ("N11", "a", "Prepare disband"),
            ("N12", "a", "Maintain"),
            ("N13", "f", "Done"),
        ],
        [
            ("N1", "N6", None),
            ("N6", "N7", "yes"),
            ("N6", "N12", "no"),
            ("N7", "N10", None),
            ("N10", "N11", "yes"),
            ("N10", "N12", "no"),
            ("N11", "N12", None),
            ("N12", "N13", None),
        ],
    )
    text = PlantUMLGenerator().render(d)
    _assert_each_action_once(text, d)
    assert text.count("stop") == 1
    assert text.splitlines()[-4:-2] == [":Maintain;", "stop"]


def test_early_stop_branch_does_not_swallow_the_rest_of_the_flow():
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N2", "d", "Malicious?"),
            ("N3", "f", "Rejected"),
            ("N4", "a", "Prepare request"),
            ("N5", "a", "Send request"),
            ("N6", "f", "Done"),
        ],
        [
            ("N1", "N2", None),
            ("N2", "N3", "yes"),
            ("N2", "N4", "no"),
            ("N4", "N5", None),
            ("N5", "N6", None),
        ],
    )
    text = PlantUMLGenerator().render(d)
    _assert_each_action_once(text, d)
    lines = text.splitlines()
    # The continuation follows the IF block instead of nesting inside it.
    assert lines.index(":Prepare request;") > lines.index("endif")


def test_fork_branches_rejoin_once():
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N2", "k", "Fork"),
            ("N3", "a", "Left"),
            ("N4", "a", "Right"),
            ("N5", "j", "Join"),
            ("N6", "a", "After"),
            ("N7", "f", "Done"),
        ],
        [
            ("N1", "N2", None),
            ("N2", "N3", None),
            ("N2", "N4", None),
            ("N3", "N5", None),
            ("N4", "N5", None),
            ("N5", "N6", None),
            ("N6", "N7", None),
        ],
    )
    text = PlantUMLGenerator().render(d)
    _assert_each_action_once(text, d)
    lines = text.splitlines()
    assert lines.index(":After;") > lines.index("end fork")


def test_single_branch_decision_compiles_instead_of_failing():
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N2", "d", "Check?"),
            ("N3", "a", "Go on"),
            ("N4", "f", "Done"),
        ],
        [("N1", "N2", None), ("N2", "N3", "ok"), ("N3", "N4", None)],
    )
    text = PlantUMLGenerator().render(d)
    assert "if (Check?) then (ok)" in text
    assert PlantUMLGenerator._blocks_balanced(text)


def test_unstructured_loop_falls_back_to_balanced_output():
    # The back edge leaves from inside a nested IF: no repeat/while block can
    # express it, so the compiler must still emit balanced PlantUML.
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N2", "a", "Head"),
            ("N3", "d", "Outer?"),
            ("N4", "d", "Inner?"),
            ("N5", "a", "Back"),
            ("N6", "a", "Side"),
            ("N7", "f", "Done"),
        ],
        [
            ("N1", "N2", None),
            ("N2", "N3", None),
            ("N3", "N4", "a"),
            ("N3", "N7", "b"),
            ("N4", "N5", "x"),
            ("N4", "N6", "y"),
            ("N5", "N2", None),
            ("N6", "N2", None),
        ],
    )
    text = PlantUMLGenerator().render(d)
    assert PlantUMLGenerator._blocks_balanced(text)


def test_unstructured_jumps_are_drawn_as_connectors():
    # Shape of outputs/Second Run/peering: N5's "no" branch continues at N7
    # (N2's merge point), but N5's own merge point is N10, so falling out
    # of N5's IF block would wrongly lead to N10. Both cross edges must be
    # kept as connector jumps, without dropping or duplicating actions.
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N2", "d", "Trigger?"),
            ("N3", "a", "Publish"),
            ("N5", "d", "Established?"),
            ("N6", "a", "Encapsulate"),
            ("N7", "a", "Receive request"),
            ("N8", "a", "Negotiate"),
            ("N10", "a", "Configure peering"),
            ("N11", "f", "Done"),
        ],
        [
            ("N1", "N2", None),
            ("N2", "N3", "yes"),
            ("N2", "N7", "no"),
            ("N3", "N5", None),
            ("N5", "N6", "yes"),
            ("N5", "N7", "no"),
            ("N6", "N10", None),
            ("N7", "N8", None),
            ("N8", "N10", None),
            ("N10", "N11", None),
        ],
    )
    text = PlantUMLGenerator().render(d)
    _assert_each_action_once(text, d)
    lines = text.splitlines()

    def target_of(connector):
        # The last occurrence marks the target; earlier ones are jumps.
        sources = [i for i, line in enumerate(lines) if line == connector]
        assert len(sources) == 2
        jump, target = sorted(sources, key=lambda i: lines[i + 1] == "detach", reverse=True)
        assert lines[jump + 1] == "detach"
        return lines[target + 1]

    targets = {target_of("(A)"), target_of("(B)")}
    assert targets == {":Receive request;", ":Configure peering;"}


def test_validator_flags_decision_whose_branches_share_one_target():
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N2", "d", "Disband?"),
            ("N3", "a", "Maintain"),
            ("N4", "f", "Done"),
        ],
        [
            ("N1", "N2", None),
            ("N2", "N3", "no"),
            ("N2", "N3", "yes"),
            ("N3", "N4", None),
        ],
    )
    ids = [defect.id for defect in StructuralValidator().validate(d).defects]
    assert "D-DECISION-003-N2" in ids


def test_composed_decision_keeps_branches_that_reach_the_same_module():
    # Module M1 ends in a decision whose two branches both finish the module.
    # Inlining must not merge them into one edge: that left the decision
    # with a single branch and the full diagram failed to compile.
    plan = DecompositionPlan(
        system_name="S",
        modules=[
            ActivityModule(id="M1", name="First", requirement_ids=["R1"]),
            ActivityModule(id="M2", name="Second", requirement_ids=["R2"]),
        ],
        transitions=[
            ModuleTransition(source="START", target="M1"),
            ModuleTransition(source="M1", target="M2"),
            ModuleTransition(source="M2", target="END"),
        ],
    )
    m1 = _diagram(
        [
            ("N1", "i", "Start"),
            ("N2", "a", "Work"),
            ("N3", "d", "Condition?"),
            ("N4", "f", "Finished A"),
            ("N5", "f", "Finished B"),
        ],
        [
            ("N1", "N2", None),
            ("N2", "N3", None),
            ("N3", "N4", "yes"),
            ("N3", "N5", "no"),
        ],
    )
    m2 = _diagram(
        [("N1", "i", "Start"), ("N2", "a", "More work"), ("N3", "f", "End")],
        [("N1", "N2", None), ("N2", "N3", None)],
    )
    full = compose_full_diagram(build_overview_diagram(plan), {"M1": m1, "M2": m2})
    outgoing = [e for e in full.edges if e.source == "M1_N3"]
    assert len(outgoing) == 2
    assert {e.guard for e in outgoing} == {"yes", "no"}
    text = PlantUMLGenerator().render(full)
    assert "if (Condition?) then (yes)" in text
    assert ":More work;" in text


def test_branches_into_an_already_drawn_final_do_not_leave_a_dangling_stop():
    # Shape of the last module in outputs/peering_*: every branch ends at the
    # shared END node, which an earlier branch already drew. A "stop" after
    # endif would be unreachable and PlantUML draws it as a loose end circle.
    d = _diagram(
        [
            ("N1", "i", "Start"),
            ("N2", "a", "Operate"),
            ("N3", "d", "Continue?"),
            ("N4", "d", "Terminate?"),
            ("N5", "d", "Exceptional?"),
            ("N6", "a", "Bypass"),
            ("N7", "a", "Disband"),
            ("END", "f", "End"),
        ],
        [
            ("N1", "N2", None),
            ("N2", "N3", None),
            ("N3", "END", "yes"),
            ("N3", "N4", "no"),
            ("N4", "N5", "yes"),
            ("N4", "END", "no"),
            ("N5", "N6", "a"),
            ("N5", "N7", "b"),
            ("N6", "END", None),
            ("N7", "END", None),
        ],
    )
    text = PlantUMLGenerator().render(d)
    _assert_each_action_once(text, d)
    lines = text.splitlines()
    after_block = [lines[i + 1] for i, line in enumerate(lines[:-1]) if line in ("endif", "end fork")]
    assert "stop" not in after_block
    assert text.count("stop") == 4
