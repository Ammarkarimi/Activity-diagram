from src.models.domain import ActivityDiagram, ActivityEdge, ActivityNode, NodeType
from src.validation.structural.structural_validator import StructuralValidator


def test_valid_simple_diagram():
    d = ActivityDiagram(
        nodes=[
            ActivityNode(id="N1", type=NodeType.INITIAL, label="Start"),
            ActivityNode(id="N2", type=NodeType.ACTION, label="Do X", requirement_ids=["R1"]),
            ActivityNode(id="N3", type=NodeType.FINAL, label="End"),
        ],
        edges=[
            ActivityEdge(id="E1", source="N1", target="N2"),
            ActivityEdge(id="E2", source="N2", target="N3"),
        ],
    )
    result = StructuralValidator().validate(d)
    assert result.passed


def test_detect_missing_final():
    d = ActivityDiagram(
        nodes=[
            ActivityNode(id="N1", type=NodeType.INITIAL, label="Start"),
            ActivityNode(id="N2", type=NodeType.ACTION, label="Do X"),
        ],
        edges=[ActivityEdge(id="E1", source="N1", target="N2")],
    )
    result = StructuralValidator().validate(d)
    assert not result.passed
