from src.generation.plantuml_generator import PlantUMLGenerator
from src.models.domain import ActivityDiagram, ActivityNode, ActivityEdge, NodeType


def test_plantuml_contains_markers():
    d = ActivityDiagram(
        nodes=[
            ActivityNode(id="N1", type=NodeType.INITIAL, label="Start"),
            ActivityNode(id="N2", type=NodeType.ACTION, label="Do X"),
            ActivityNode(id="N3", type=NodeType.FINAL, label="End"),
        ],
        edges=[
            ActivityEdge(id="E1", source="N1", target="N2"),
            ActivityEdge(id="E2", source="N2", target="N3"),
        ],
    )
    text = PlantUMLGenerator().render(d)
    assert text.startswith("@startuml")
    assert text.endswith("@enduml")


def test_plantuml_multi_branch_decision():
    d = ActivityDiagram(
        nodes=[
            ActivityNode(id="N1", type=NodeType.INITIAL, label="Start"),
            ActivityNode(id="N2", type=NodeType.DECISION, label="Check status"),
            ActivityNode(id="N3", type=NodeType.ACTION, label="Action A"),
            ActivityNode(id="N4", type=NodeType.ACTION, label="Action B"),
            ActivityNode(id="N5", type=NodeType.ACTION, label="Action C"),
            ActivityNode(id="N6", type=NodeType.FINAL, label="End"),
        ],
        edges=[
            ActivityEdge(id="E1", source="N1", target="N2"),
            ActivityEdge(id="E2", source="N2", target="N3", guard="status == A"),
            ActivityEdge(id="E3", source="N2", target="N4", guard="status == B"),
            ActivityEdge(id="E4", source="N2", target="N5", guard="otherwise"),
            ActivityEdge(id="E5", source="N3", target="N6"),
            ActivityEdge(id="E6", source="N4", target="N6"),
            ActivityEdge(id="E7", source="N5", target="N6"),
        ],
    )
    text = PlantUMLGenerator().render(d)
    assert "if (" in text
    assert "elseif (" in text
    assert "else (" in text
    assert "endif" in text

    # Verify that regex validator passes it
    from src.validation.syntax.plantuml_validator import PlantUMLSyntaxValidator
    valid, msg = PlantUMLSyntaxValidator()._regex_validate(text)
    assert valid, f"Regex validation failed: {msg}"


def test_plantuml_special_characters_escaping():
    d = ActivityDiagram(
        nodes=[
            ActivityNode(id="N1", type=NodeType.INITIAL, label="Start"),
            ActivityNode(id="N2", type=NodeType.ACTION, label='Do "X"; {special} | <tag>'),
            ActivityNode(id="N3", type=NodeType.FINAL, label="End"),
        ],
        edges=[
            ActivityEdge(id="E1", source="N1", target="N2"),
            ActivityEdge(id="E2", source="N2", target="N3"),
        ],
    )
    text = PlantUMLGenerator().render(d)
    assert 'Do \'X\',' in text
    assert '{' not in text and '}' not in text
    assert '|' not in text


def test_ir_sanitizer():
    from src.generation.ir_sanitizer import IRSanitizer
    d = ActivityDiagram(
        title="",
        nodes=[
            ActivityNode(id="N 1", type=NodeType.INITIAL, label="Start"),
            ActivityNode(id="N 1", type=NodeType.INITIAL, label="Duplicate Start"),
            ActivityNode(id="N 2", type=NodeType.FINAL, label="End"),
        ],
        edges=[
            ActivityEdge(id="E 1", source="N 1", target="N 2", guard=""),
            ActivityEdge(id="E 2", source="N 1", target="N_GHOST"),  # dangling
        ],
    )
    sanitized = IRSanitizer.sanitize(d)
    assert sanitized.title == "Activity Diagram"
    assert len(sanitized.nodes) == 2  # duplicate removed
    assert sanitized.nodes[0].id == "N_1"
    assert sanitized.nodes[1].id == "N_2"
    assert len(sanitized.edges) == 1  # dangling removed
    assert sanitized.edges[0].source == "N_1"
    assert sanitized.edges[0].target == "N_2"
    assert sanitized.edges[0].guard is None  # empty string converted to None

