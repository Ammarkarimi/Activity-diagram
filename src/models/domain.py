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

    type: RequirementType = RequirementType.ACTION

    actors: list[str] = Field(
        default_factory=list
    )

    actions: list[str] = Field(
        default_factory=list
    )

    conditions: list[str] = Field(
        default_factory=list
    )

    exceptions: list[str] = Field(
        default_factory=list
    )

    dependencies: list[str] = Field(
        default_factory=list
    )

    source_sentence: str = ""


class RequirementMatrixItem(BaseModel):
    requirement_id: str

    diagram_node_ids: list[str] = Field(
        default_factory=list
    )

    diagram_edge_ids: list[str] = Field(
        default_factory=list
    )

    evidence: str = ""

    confidence: float = 0.0


class RequirementMatrix(BaseModel):
    items: list[RequirementMatrixItem] = Field(
        default_factory=list
    )


# ============================================================
# PLAN MODELS
# ============================================================


class PlanNode(BaseModel):
    temp_id: str

    node_type: NodeType

    label: str

    requirement_ids: list[str] = Field(
        default_factory=list
    )


class PlanEdge(BaseModel):
    source: str

    target: str

    guard: str | None = None

    edge_type: EdgeType = EdgeType.CONTROL

    requirement_ids: list[str] = Field(
        default_factory=list
    )


class DecisionPlan(BaseModel):
    decision_id: str

    condition: str

    # IDs of outgoing branch targets.
    branch_ids: list[str] = Field(
        default_factory=list
    )

    # Guard corresponding to each branch.
    guards: list[str] = Field(
        default_factory=list
    )


class LoopPlan(BaseModel):
    loop_id: str

    condition: str

    # Node that starts the loop body.
    entry_node: str

    # Nodes executed inside the loop.
    body_nodes: list[str] = Field(
        default_factory=list
    )

    # Node after loop exits.
    exit_node: str

    # Conditions for continuing/exiting.
    continue_guard: str

    exit_guard: str


class ConcurrencyPlan(BaseModel):
    fork_id: str

    # First nodes of parallel branches.
    branch_ids: list[str] = Field(
        default_factory=list
    )

    join_id: str


class ExceptionPlan(BaseModel):
    exception_id: str

    trigger: str

    handler_node_id: str

    requirement_ids: list[str] = Field(
        default_factory=list
    )


class ActivityPlan(BaseModel):
    objective: str

    nodes: list[PlanNode] = Field(
        default_factory=list
    )

    edges: list[PlanEdge] = Field(
        default_factory=list
    )

    decisions: list[DecisionPlan] = Field(
        default_factory=list
    )

    loops: list[LoopPlan] = Field(
        default_factory=list
    )

    concurrency: list[ConcurrencyPlan] = Field(
        default_factory=list
    )

    exceptions: list[ExceptionPlan] = Field(
        default_factory=list
    )

    decision_strategy: list[str] = Field(
        default_factory=list
    )

    concurrency_strategy: list[str] = Field(
        default_factory=list
    )

    exception_strategy: list[str] = Field(
        default_factory=list
    )

    termination_strategy: list[str] = Field(
        default_factory=list
    )

    traceability: RequirementMatrix = Field(
        default_factory=RequirementMatrix
    )


# ============================================================
# ACTIVITY DIAGRAM IR
# ============================================================


class ActivityNode(BaseModel):
    id: str

    type: NodeType

    label: str

    requirement_ids: list[str] = Field(
        default_factory=list
    )


class ActivityEdge(BaseModel):
    id: str

    source: str

    target: str

    type: EdgeType = EdgeType.CONTROL

    guard: str | None = None

    requirement_ids: list[str] = Field(
        default_factory=list
    )


class ActivityDiagram(BaseModel):
    nodes: list[ActivityNode] = Field(
        default_factory=list
    )

    edges: list[ActivityEdge] = Field(
        default_factory=list
    )

    title: str = "Activity Diagram"

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

    node_ids: list[str] = Field(
        default_factory=list
    )

    edge_ids: list[str] = Field(
        default_factory=list
    )

    requirement_ids: list[str] = Field(
        default_factory=list
    )

    evidence: str = ""

    suggested_action: str = ""


class ValidationResult(BaseModel):
    passed: bool

    score: float = 0.0

    defects: list[Defect] = Field(
        default_factory=list
    )

    # Counts of deterministic validation rules.
    #
    # Example:
    #
    # {
    #     "total": 10,
    #     "passed": 9,
    #     "failed": 1
    # }
    rule_counts: dict[str, int] = Field(
        default_factory=dict
    )

    summary: str = ""


# ============================================================
# REVIEW MODELS
# ============================================================


class ReviewResult(BaseModel):
    overall_assessment: str = ""

    semantic_score: float = 0.0

    defects: list[Defect] = Field(
        default_factory=list
    )

    strengths: list[str] = Field(
        default_factory=list
    )

    missing_requirements: list[str] = Field(
        default_factory=list
    )

    unsupported_behaviors: list[str] = Field(
        default_factory=list
    )


# ============================================================
# REPAIR MODELS
# ============================================================


class RepairResult(BaseModel):
    changed: bool = False

    repair_type: str = "NONE"

    changes: list[str] = Field(
        default_factory=list
    )

    rationale: str = ""

    diagram: ActivityDiagram


# ============================================================
# HUMAN EVALUATION
# ============================================================


class HumanRating(BaseModel):
    """
    Human evaluation for one generated activity diagram.

    Ratings should normally use a consistent scale such as
    1-5 for all dimensions.
    """

    sample_id: str

    rater_id: str

    semantic_correctness: int

    completeness: int

    structural_correctness: int

    readability: int

    abstraction: int

    overall_quality: int


# ============================================================
# REQUIREMENT COVERAGE
# ============================================================


class RequirementCoverage(BaseModel):
    """
    Measures whether an individual requirement is represented
    in the generated diagram.
    """

    requirement_id: str

    covered: bool = False

    coverage_type: str = "NONE"

    evidence: str = ""

    confidence: float = 0.0


# ============================================================
# CANDIDATE-LEVEL METRICS
# ============================================================


class CandidateMetrics(BaseModel):
    """
    Evaluation metrics for one diagram candidate.

    These values allow us to compare candidates produced at
    different repair iterations.
    """

    # --------------------------------------------------------
    # Iteration
    # --------------------------------------------------------

    iteration: int

    # --------------------------------------------------------
    # Requirement-level metrics
    # --------------------------------------------------------

    requirement_coverage: float = 0.0

    correctness: float = 0.0

    completeness: float = 0.0

    # --------------------------------------------------------
    # Structural metrics
    # --------------------------------------------------------

    structural_validity: float = 0.0

    # --------------------------------------------------------
    # Semantic metrics
    # --------------------------------------------------------

    unsupported_behaviour_rate: float = 0.0

    # --------------------------------------------------------
    # Defect metrics
    # --------------------------------------------------------

    defect_count: int = 0

    defects_fixed: int = 0

    defects_introduced: int = 0

    defect_reduction_rate: float = 0.0

    # --------------------------------------------------------
    # Repair metrics
    # --------------------------------------------------------

    repair_success_rate: float = 0.0

    # --------------------------------------------------------
    # Overall candidate quality
    # --------------------------------------------------------

    quality_score: float = 0.0

    # --------------------------------------------------------
    # Efficiency metrics
    # --------------------------------------------------------

    execution_time_seconds: float = 0.0

    llm_calls: int = 0


# ============================================================
# CANDIDATE RECORD
# ============================================================


class CandidateRecord(BaseModel):
    """
    Complete snapshot of a candidate produced during one
    pipeline iteration.
    """

    iteration: int

    diagram: ActivityDiagram

    plantuml: str = ""

    metrics: CandidateMetrics

    defects: list[Defect] = Field(
        default_factory=list
    )

    requirement_coverage: list[
        RequirementCoverage
    ] = Field(
        default_factory=list
    )


# ============================================================
# PIPELINE STATE
# ============================================================


class PipelineState(BaseModel):
    """
    Complete state of one pipeline execution.

    This object is also the main source for experimental
    evaluation and ablation analysis.
    """

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    sample_id: str

    requirement_text: str

    # --------------------------------------------------------
    # Requirement analysis
    # --------------------------------------------------------

    requirements: list[Requirement] = Field(
        default_factory=list
    )

    requirement_matrix: RequirementMatrix = Field(
        default_factory=RequirementMatrix
    )

    # --------------------------------------------------------
    # Planning
    # --------------------------------------------------------

    plan: ActivityPlan | None = None

    # --------------------------------------------------------
    # Current diagram
    # --------------------------------------------------------

    diagram: ActivityDiagram | None = None

    # --------------------------------------------------------
    # Current validation / review
    # --------------------------------------------------------

    validation: ValidationResult | None = None

    review: ReviewResult | None = None

    # --------------------------------------------------------
    # Current defects
    # --------------------------------------------------------

    defects: list[Defect] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Repair history
    # --------------------------------------------------------

    repair_history: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Iteration tracking
    # --------------------------------------------------------

    iteration: int = 0

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    final_plantuml: str = ""

    # --------------------------------------------------------
    # General metrics
    # --------------------------------------------------------

    metrics: dict[str, Any] = Field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Candidate tracking
    # --------------------------------------------------------

    candidates: list[CandidateRecord] = Field(
        default_factory=list
    )

    best_candidate_iteration: int | None = None

    best_candidate: CandidateRecord | None = None


# ============================================================
# EXPERIMENT / EVALUATION METRICS
# ============================================================


class PipelineEvaluation(BaseModel):
    """
    Dataset-level evaluation summary.

    This is useful when running the system over multiple
    requirements and comparing different configurations.
    """

    total_samples: int = 0

    successful_samples: int = 0

    failed_samples: int = 0

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    average_correctness: float = 0.0

    average_completeness: float = 0.0

    average_requirement_coverage: float = 0.0

    average_structural_validity: float = 0.0

    average_quality_score: float = 0.0

    # --------------------------------------------------------
    # Defects
    # --------------------------------------------------------

    average_initial_defects: float = 0.0

    average_final_defects: float = 0.0

    average_defect_reduction_rate: float = 0.0

    # --------------------------------------------------------
    # Iterative behaviour
    # --------------------------------------------------------

    average_iterations: float = 0.0

    average_best_iteration: float = 0.0

    # --------------------------------------------------------
    # Repair behaviour
    # --------------------------------------------------------

    average_repair_success_rate: float = 0.0

    # --------------------------------------------------------
    # Efficiency
    # --------------------------------------------------------

    average_execution_time_seconds: float = 0.0

    average_llm_calls: float = 0.0


# ============================================================
# ABLATION CONFIGURATION
# ============================================================


class PipelineConfiguration(BaseModel):
    """
    Configuration used for one experimental run.

    This allows the complete system to later be compared
    against ablated versions.
    """

    use_requirement_agent: bool = True

    use_planning_agent: bool = True

    use_generator_agent: bool = True

    use_deterministic_validation: bool = True

    use_semantic_review: bool = True

    use_feedback_agent: bool = True

    use_repair_router: bool = True

    use_deterministic_repairs: bool = True

    use_candidate_selection: bool = True

    max_iterations: int = 3

    max_defects_per_repair: int = 2


# ============================================================
# EXPERIMENT RESULT
# ============================================================


class ExperimentResult(BaseModel):
    """
    Stores the result of one experimental configuration.
    """

    configuration_name: str

    configuration: PipelineConfiguration

    evaluation: PipelineEvaluation

    sample_results: list[PipelineState] = Field(
        default_factory=list
    )