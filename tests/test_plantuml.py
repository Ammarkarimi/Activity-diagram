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
