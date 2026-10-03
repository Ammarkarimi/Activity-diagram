"""PlantUML layout crashes and the plain-branching fallback.

PlantUML 1.2024.7 crashes (NullPointerException in FtileWhile) on a while loop
whose body ends in another swimlane with a jump or a stop, as in
outputs/peering_20261003_105804_795973/full.puml. It still exits with 0 and
returns an image of the error, which used to be reported as a valid diagram.
"""
import logging

import pytest

from src.generation.plantuml_generator import PlantUMLGenerator
from src.generation.plantuml_tool import plantuml_command
from src.models.domain import ActivityDiagram, ActivityEdge, ActivityNode, NodeType
from src.pipeline.hierarchical import HierarchicalPipeline
from src.pipeline.orchestrator import MultiAgentPipeline
from src.validation.syntax.plantuml_validator import PlantUMLCheck, PlantUMLSyntaxValidator

CRASHING = """@startuml
|A|
start
while (c?) is (yes)
:x;
|B|
:y;
stop
endwhile (no)
|A|
:z;
stop
@enduml
"""


def _loop_diagram():
    # Decision at the loop head: compiled as a while loop by default.
    kinds = {"i": NodeType.INITIAL, "a": NodeType.ACTION, "d": NodeType.DECISION, "f": NodeType.FINAL}
    nodes = [
        ("N1", "i", "Start"),
        ("N3", "a", "Advertise"),
        ("N4", "d", "Established?"),
        ("N5", "a", "Exchange"),
        ("N11", "a", "Re-negotiate"),
        ("N13", "f", "Done"),
    ]
    edges = [("N1", "N3", None), ("N3", "N4", None), ("N4", "N5", "yes"),
             ("N4", "N11", "no"), ("N11", "N4", None), ("N5", "N13", None)]
    return ActivityDiagram(
        title="T",
        nodes=[ActivityNode(id=i, type=kinds[k], label=label) for i, k, label in nodes],
        edges=[ActivityEdge(id=f"E{n}", source=s, target=t, guard=g) for n, (s, t, g) in enumerate(edges, 1)],
    )


class _RejectWhile:
    """Stands in for PlantUML: 'crashes' on any while loop."""

    def check(self, text):
        if "\nwhile (" in text:
            return PlantUMLCheck(valid=False, message="PlantUML failed to draw the diagram: NPE", verified=True)
        return PlantUMLCheck(valid=True, message="ok", verified=True)

    def validate(self, text):
        result = self.check(text)
        return result.valid, result.message


def test_plain_loops_draw_no_loop_blocks_and_keep_every_action():
    diagram = _loop_diagram()
    assert "\nwhile (" in PlantUMLGenerator().render(diagram)
    text = PlantUMLGenerator().render(diagram, structured_loops=False)
    assert "while (" not in text and "repeat" not in text
    for label in ("Advertise", "Exchange", "Re-negotiate"):
        assert text.count(f":{label};") == 1


@pytest.mark.skipif(plantuml_command() is None, reason="PlantUML not installed")
def test_validator_reports_layout_crash_as_invalid():
    result = PlantUMLSyntaxValidator().check(CRASHING)
    assert not result.valid and result.verified
    assert "failed to draw" in result.message


def test_orchestrator_finalises_plain_branching_when_structured_loops_fail():
    pipeline = MultiAgentPipeline.__new__(MultiAgentPipeline)
    pipeline.plantuml = PlantUMLGenerator()
    pipeline.plantuml_validator = _RejectWhile()
    pipeline.log = logging.getLogger("test")
    text, defect = pipeline._compile_plantuml(_loop_diagram(), iteration=0)
    assert defect is None
    assert "while (" not in text and ":Re-negotiate;" in text


def test_hierarchical_check_retries_with_plain_branching():
    pipeline = HierarchicalPipeline.__new__(HierarchicalPipeline)
    pipeline.syntax = _RejectWhile()
    pipeline.log = logging.getLogger("test")
    diagram = _loop_diagram()
    text = PlantUMLGenerator().render(diagram)
    final, result = pipeline._plain_loops_if_invalid(diagram, text, pipeline.syntax.check(text), "full")
    assert result.valid and "while (" not in final


def test_hierarchical_keeps_original_when_fallback_also_fails():
    class _RejectAll(_RejectWhile):
        def check(self, text):
            return PlantUMLCheck(valid=False, message="bad", verified=True)

    pipeline = HierarchicalPipeline.__new__(HierarchicalPipeline)
    pipeline.syntax = _RejectAll()
    pipeline.log = logging.getLogger("test")
    diagram = _loop_diagram()
    text = PlantUMLGenerator().render(diagram)
    final, result = pipeline._plain_loops_if_invalid(diagram, text, pipeline.syntax.check(text), "full")
    assert final == text and not result.valid
