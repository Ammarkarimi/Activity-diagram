from src.evaluation.graph_metrics import normalized_graph_edit_similarity
from src.evaluation.requirement_metrics import requirement_coverage
from src.models.domain import ActivityDiagram, ActivityNode, NodeType, Requirement


def test_graph_similarity_identical():
    d = ActivityDiagram(
        nodes=[
            ActivityNode(id="N1", type=NodeType.INITIAL, label="Start"),
            ActivityNode(id="N2", type=NodeType.FINAL, label="End"),
        ]
    )
    assert normalized_graph_edit_similarity(d, d) == 1.0


def test_requirement_coverage():
    d = ActivityDiagram(
        nodes=[ActivityNode(id="N1", type=NodeType.ACTION, label="X", requirement_ids=["R1"])]
    )
    reqs = [Requirement(id="R1", text="Do X")]
    assert requirement_coverage(d, reqs) == 1.0
