from pathlib import Path

from src.generation.plantuml_generator import PlantUMLGenerator
from src.models.domain import NodeType
from src.parsing.activity_parser import ActivityPlantUMLParser
from src.validation.structural.structural_validator import StructuralValidator
from tests.test_long_document import _module_diagram


def test_parses_gold_standards_into_valid_graphs():
    for path in Path("Gold Standard Solutions").glob("*.txt"):
        diagram = ActivityPlantUMLParser().parse(path.read_text(encoding="utf-8"))
        assert sum(n.type == NodeType.ACTION for n in diagram.nodes) >= 10, path
        assert sum(n.type == NodeType.INITIAL for n in diagram.nodes) == 1, path
        assert StructuralValidator().validate(diagram).score >= 0.9, path
        assert any(n.lane for n in diagram.nodes), path


def test_round_trip_of_generated_plantuml():
    original = _module_diagram("A", with_reject=True)
    for n in original.nodes:
        n.lane = "Clerk"
    parsed = ActivityPlantUMLParser().parse(PlantUMLGenerator().render(original))
    labels = {n.label for n in parsed.nodes if n.type == NodeType.ACTION}
    assert labels == {"A work"}
    assert sum(n.type == NodeType.DECISION for n in parsed.nodes) == 1
    assert sum(n.type == NodeType.FINAL for n in parsed.nodes) == 2
    assert {n.lane for n in parsed.nodes if n.type == NodeType.ACTION} == {"Clerk"}
