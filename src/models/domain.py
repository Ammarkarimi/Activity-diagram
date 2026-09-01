from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# ENUMS
# ============================================================

class RequirementType(str, Enum):
    ACTION = "ACTION"
    CONDITION = "CONDITION"
    DECISION = "DECISION"
    LOOP = "LOOP"
    CONCURRENCY = "CONCURRENCY"
    EXCEPTION = "EXCEPTION"
    DATA = "DATA"
    TERMINATION = "TERMINATION"
    ACTOR = "ACTOR"
    OTHER = "OTHER"


class NodeType(str, Enum):
    INITIAL = "initial"
    FINAL = "final"
    ACTION = "action"
    DECISION = "decision"
    MERGE = "merge"
    FORK = "fork"
    JOIN = "join"
    OBJECT = "object"
    NOTE = "note"


class EdgeType(str, Enum):
    CONTROL = "control"
    OBJECT = "object"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================
# REQUIREMENT MODELS
# ============================================================

class Requirement(BaseModel):
    id: str
    text: str
    type: RequirementType
    actors: list[str]
    actions: list[str]
    conditions: list[str]
    exceptions: list[str]
    dependencies: list[str]
    source_sentence: str


class RequirementMatrixItem(BaseModel):
    requirement_id: str
    diagram_node_ids: list[str]
    diagram_edge_ids: list[str]
    evidence: str
    confidence: float


class RequirementMatrix(BaseModel):
    items: list[RequirementMatrixItem]


# ============================================================
# PLAN MODELS
# ============================================================

class PlanNode(BaseModel):
    temp_id: str
    node_type: NodeType
    label: str
    requirement_ids: list[str]


class PlanEdge(BaseModel):
    source: str
    target: str
    guard: str | None
    edge_type: EdgeType
    requirement_ids: list[str]


class DecisionPlan(BaseModel):
    decision_id: str
    condition: str

    # IDs of outgoing branch targets.
    branch_ids: list[str]

    # Guard corresponding to each branch_id.
    guards: list[str]


class LoopPlan(BaseModel):
    loop_id: str
    condition: str

    # Node that starts the loop body.
    entry_node: str

    # Nodes executed inside loop.
    body_nodes: list[str]

    # Node after loop exits.
    exit_node: str

    # Conditions for continuing/exiting.
    continue_guard: str
    exit_guard: str


class ConcurrencyPlan(BaseModel):
    fork_id: str

    # First nodes of parallel branches.
    branch_ids: list[str]

    join_id: str


class ExceptionPlan(BaseModel):
    exception_id: str
    trigger: str
    handler_node_id: str
    requirement_ids: list[str]


class ActivityPlan(BaseModel):
    objective: str

    nodes: list[PlanNode]
    edges: list[PlanEdge]

    decisions: list[DecisionPlan]
    loops: list[LoopPlan]
    concurrency: list[ConcurrencyPlan]
    exceptions: list[ExceptionPlan]

    decision_strategy: list[str]
    concurrency_strategy: list[str]
    exception_strategy: list[str]
    termination_strategy: list[str]

    traceability: RequirementMatrix


# ============================================================
# ACTIVITY DIAGRAM IR
# ============================================================

class ActivityNode(BaseModel):
    id: str
    type: NodeType
    label: str
    requirement_ids: list[str]


class ActivityEdge(BaseModel):
    id: str
    source: str
    target: str
    type: EdgeType
    guard: str | None
    requirement_ids: list[str]


class ActivityDiagram(BaseModel):
    nodes: list[ActivityNode]
    edges: list[ActivityEdge]
    title: str

    def node_map(self) -> dict[str, ActivityNode]:
        return {
            node.id: node
            for node in self.nodes
        }

    def edge_map(self) -> dict[str, ActivityEdge]:
        return {
            edge.id: edge
            for edge in self.edges
        }


# ============================================================
# DEFECT / VALIDATION MODELS
# ============================================================

class Defect(BaseModel):
    id: str
    category: str
    severity: Severity

    description: str

    node_ids: list[str]
    edge_ids: list[str]
    requirement_ids: list[str]

    evidence: str
    suggested_action: str


class ValidationResult(BaseModel):
    passed: bool
    score: float

    defects: list[Defect]

    rule_counts: dict[str, int]

    summary: str


# ============================================================
# REVIEW MODELS
# ============================================================

class ReviewResult(BaseModel):
    overall_assessment: str

    semantic_score: float

    defects: list[Defect]

    strengths: list[str]

    missing_requirements: list[str]

    unsupported_behaviors: list[str]


# ============================================================
# REPAIR MODELS
# ============================================================

class RepairResult(BaseModel):
    changed: bool

    repair_type: str

    changes: list[str]

    rationale: str

    diagram: ActivityDiagram


# ============================================================
# HUMAN EVALUATION
# ============================================================

class HumanRating(BaseModel):
    sample_id: str
    rater_id: str

    semantic_correctness: int
    completeness: int
    structural_correctness: int
    readability: int
    abstraction: int
    overall_quality: int


# ============================================================
# PIPELINE STATE
# ============================================================

class PipelineState(BaseModel):
    sample_id: str
    requirement_text: str

    requirements: list[Requirement]
    requirement_matrix: RequirementMatrix

    plan: ActivityPlan | None
    diagram: ActivityDiagram | None

    validation: ValidationResult | None
    review: ReviewResult | None

    defects: list[Defect]

    repair_history: list[dict[str, Any]]

    iteration: int

    final_plantuml: str

    metrics: dict[str, Any]