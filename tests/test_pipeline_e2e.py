from __future__ import annotations

from unittest.mock import MagicMock
from src.generation.plantuml_generator import PlantUMLGenerator
from src.generation.ir_sanitizer import IRSanitizer
from src.models.domain import (
    ActivityDiagram,
    ActivityEdge,
    ActivityNode,
    ActivityPlan,
    DecisionPlan,
    NodeType,
    EdgeType,
    PlanEdge,
    PlanNode,
    Requirement,
    RequirementMatrix,
    RequirementType,
    ReviewResult,
)
from src.validation.syntax.plantuml_validator import PlantUMLSyntaxValidator


def test_complete_pipeline_flow():
    reqs = [
        Requirement(
            id='R1',
            text='Customer enters credentials.',
            type=RequirementType.ACTION,
            actions=['enter credentials'],
            source_sentence='Customer enters credentials.',
        ),
        Requirement(
            id='R2',
            text='System validates credentials.',
            type=RequirementType.ACTION,
            actions=['validate credentials'],
            source_sentence='System validates credentials.',
        ),
        Requirement(
            id='R3',
            text='If valid, show dashboard; if invalid, show error and stop.',
            type=RequirementType.DECISION,
            conditions=['valid', 'invalid'],
            source_sentence='If valid, show dashboard; if invalid, show error and stop.',
        ),
    ]

    generated_raw = ActivityDiagram(
        title='Login Flow',
        nodes=[
            ActivityNode(id='N 1', type=NodeType.INITIAL, label='Start', requirement_ids=[]),
            ActivityNode(id='N 2', type=NodeType.ACTION, label='Enter credentials', requirement_ids=['R1']),
            ActivityNode(id='N 3', type=NodeType.DECISION, label='Credentials valid?', requirement_ids=['R2']),
            ActivityNode(id='N 4', type=NodeType.ACTION, label='Open dashboard', requirement_ids=['R3']),
            ActivityNode(id='N 5', type=NodeType.ACTION, label='Display error', requirement_ids=['R3']),
            ActivityNode(id='N 6', type=NodeType.FINAL, label='Success End', requirement_ids=[]),
            ActivityNode(id='N 7', type=NodeType.FINAL, label='Error End', requirement_ids=[]),
        ],
        edges=[
            ActivityEdge(id='E 1', source='N 1', target='N 2', type=EdgeType.CONTROL, guard=None, requirement_ids=[]),
            ActivityEdge(id='E 2', source='N 2', target='N 3', type=EdgeType.CONTROL, guard=None, requirement_ids=['R1']),
            ActivityEdge(id='E 3', source='N 3', target='N 4', type=EdgeType.CONTROL, guard='valid', requirement_ids=['R3']),
            ActivityEdge(id='E 4', source='N 3', target='N 5', type=EdgeType.CONTROL, guard='invalid', requirement_ids=['R3']),
            ActivityEdge(id='E 5', source='N 4', target='N 6', type=EdgeType.CONTROL, guard=None, requirement_ids=[]),
            ActivityEdge(id='E 6', source='N 5', target='N 7', type=EdgeType.CONTROL, guard=None, requirement_ids=[]),
            ActivityEdge(id='E_DANGLING', source='N 1', target='NON_EXISTENT', type=EdgeType.CONTROL, guard=None, requirement_ids=[]),
        ],
    )

    sanitized = IRSanitizer.sanitize(generated_raw)
    assert len(sanitized.edges) == 6
    assert all(' ' not in n.id for n in sanitized.nodes)
    assert all(' ' not in e.id for e in sanitized.edges)

    puml = PlantUMLGenerator().render(sanitized)
    assert '@startuml' in puml
    assert '@enduml' in puml
    assert 'if (' in puml
    assert 'else (' in puml
    assert 'endif' in puml
    assert 'stop' in puml

    validator = PlantUMLSyntaxValidator()
    valid, msg = validator._regex_validate(puml)
    assert valid, f'Syntax validator failed: {msg}'

    from src.repair.deterministic_repairs import StructuralRepair
    missing_final_diagram = ActivityDiagram(
        title='Test',
        nodes=[
            ActivityNode(id='N1', type=NodeType.INITIAL, label='Start', requirement_ids=[]),
            ActivityNode(id='N2', type=NodeType.ACTION, label='Step 1', requirement_ids=['R1']),
        ],
        edges=[
            ActivityEdge(id='E1', source='N1', target='N2', type=EdgeType.CONTROL, guard=None, requirement_ids=['R1']),
        ],
    )
    rep_result = StructuralRepair().repair(missing_final_diagram, [])
    assert rep_result.changed
    assert any(n.type == NodeType.FINAL for n in rep_result.diagram.nodes)
    final_edge = rep_result.diagram.edges[-1]
    assert final_edge.type == EdgeType.CONTROL
    assert final_edge.requirement_ids == []
