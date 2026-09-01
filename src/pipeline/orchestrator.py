from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agents.feedback_agent import (
    FeedbackAgent,
)
from src.agents.generator_agent import (
    GeneratorAgent,
)
from src.agents.planning_agent import (
    PlanningAgent,
)
from src.agents.requirement_agent import (
    RequirementAgent,
)
from src.agents.reviewer_agent import (
    ReviewerAgent,
)
from src.generation.plantuml_generator import (
    PlantUMLGenerator,
)
from src.generation.renderer import (
    PlantUMLRenderer,
)
from src.llm.openai_client import (
    OpenAIClient,
)
from src.models.domain import (
    ActivityDiagram,
    Defect,
    PipelineState,
    RequirementMatrix,
    ReviewResult,
    Severity,
    ValidationResult,
)
from src.pipeline.repair_router import (
    RepairRouter,
)
from src.validation.syntax.plantuml_validator import (
    PlantUMLSyntaxValidator,
)
from src.validation.validator import (
    HybridValidator,
)


class MultiAgentPipeline:
    """
    Complete requirement -> activity diagram pipeline.

    Pipeline:

        Requirement
            |
            v
        Requirement Agent
            |
            v
        Requirement Matrix
            |
            v
        Planning Agent
            |
            v
        Activity Plan
            |
            v
        Generator Agent
            |
            v
        Activity Diagram IR
            |
            +-------------------------------+
            |                               |
            v                               v
      Deterministic Validator          Semantic Reviewer
            |                               |
            +---------------+---------------+
                            |
                            v
                      Defect Set
                            |
                            v
                      Repair Router
                            |
                            v
                     New Candidate
                            |
                            v
                       Regression
                            |
                  +---------+---------+
                  |                   |
                Better              Worse
                  |                   |
                  v                   v
              Keep                  Rollback
                  |
                  v
             Best Candidate
                  |
                  v
             PlantUML Compiler
                  |
                  v
          Final PlantUML / Rendering
    """

    def __init__(
        self,
        model: str | None = None,
        max_defects_per_repair: int = 2,
    ) -> None:

        self.llm = OpenAIClient(
            model=model
        )

        # --------------------------------------------------------
        # Agents
        # --------------------------------------------------------

        self.requirement_agent = (
            RequirementAgent(
                self.llm
            )
        )

        self.planning_agent = (
            PlanningAgent(
                self.llm
            )
        )

        self.generator = (
            GeneratorAgent(
                self.llm
            )
        )

        self.reviewer = (
            ReviewerAgent(
                self.llm
            )
        )

        self.feedback = (
            FeedbackAgent(
                self.llm
            )
        )

        # --------------------------------------------------------
        # Pipeline components
        # --------------------------------------------------------

        self.repair_router = (
            RepairRouter(
                self.llm,
                max_defects_per_repair=(
                    max_defects_per_repair
                ),
            )
        )

        self.validator = (
            HybridValidator()
        )

        self.plantuml = (
            PlantUMLGenerator()
        )

        self.plantuml_validator = (
            PlantUMLSyntaxValidator()
        )

        self.renderer = (
            PlantUMLRenderer()
        )

        self.log = logging.getLogger(
            self.__class__.__name__
        )

    # ============================================================
    # MAIN PIPELINE
    # ============================================================

    def run(
        self,
        sample_id: str,
        requirement_text: str,
        max_iterations: int = 3,
        output_dir: str | Path = "outputs",
    ) -> PipelineState:

        # --------------------------------------------------------
        # Initial State
        # --------------------------------------------------------

        state = PipelineState(
            sample_id=sample_id,
            requirement_text=(
                requirement_text
            ),
            requirements=[],
            requirement_matrix=(
                RequirementMatrix(
                    items=[]
                )
            ),
            plan=None,
            diagram=None,
            validation=None,
            review=None,
            defects=[],
            repair_history=[],
            iteration=0,
            final_plantuml="",
            metrics={},
        )

        # ========================================================
        # PHASE 1 — REQUIREMENT EXTRACTION
        # ========================================================

        self.log.info(
            "Running requirement agent..."
        )

        extracted = (
            self.requirement_agent.run(
                requirement_text
            )
        )

        state.requirements = (
            extracted.requirements
        )

        state.requirement_matrix = (
            extracted.matrix
        )

        self.log.info(
            "Extracted %d requirements.",
            len(
                state.requirements
            ),
        )

        # ========================================================
        # PHASE 2 — PLANNING
        # ========================================================

        self.log.info(
            "Running planning agent..."
        )

        state.plan = (
            self.planning_agent.run(
                requirement_text,
                state.requirements,
                state.requirement_matrix,
            )
        )

        self.log.info(
            "Planning completed: "
            "%d planned nodes, "
            "%d planned edges, "
            "%d decisions, "
            "%d loops, "
            "%d concurrency blocks.",
            len(
                state.plan.nodes
            ),
            len(
                state.plan.edges
            ),
            len(
                state.plan.decisions
            ),
            len(
                state.plan.loops
            ),
            len(
                state.plan.concurrency
            ),
        )

        # ========================================================
        # PHASE 3 — INITIAL GENERATION
        # ========================================================

        self.log.info(
            "Running generator agent..."
        )

        state.diagram = (
            self.generator.run(
                requirement_text,
                state.requirements,
                state.plan,
            )
        )

        self.log.info(
            "Generated diagram: "
            "%d nodes, %d edges.",
            len(
                state.diagram.nodes
            ),
            len(
                state.diagram.edges
            ),
        )

        # ========================================================
        # BEST CANDIDATE TRACKING
        # ========================================================

        best_diagram = (
            state.diagram.model_copy(
                deep=True
            )
        )

        best_validation: (
            ValidationResult | None
        ) = None

        best_review: (
            ReviewResult | None
        ) = None

        best_score = float(
            "-inf"
        )

        best_iteration = 0

        # ========================================================
        # ITERATIVE VALIDATION / REPAIR
        # ========================================================

        for iteration in range(
            max_iterations + 1
        ):

            state.iteration = (
                iteration
            )

            self.log.info(
                "Validation iteration %d",
                iteration,
            )

            # ====================================================
            # 1. DETERMINISTIC VALIDATION
            # ====================================================

            validation = (
                self.validator.validate(
                    state.diagram,
                    state.requirements,
                )
            )

            state.validation = (
                validation
            )

            defects: list[Defect] = list(
                validation.defects
            )

            self.log.info(
                "Deterministic validation "
                "found %d defect(s).",
                len(
                    validation.defects
                ),
            )

            # ====================================================
            # 2. COMPILE TO PLANTUML
            # ====================================================

            candidate_plantuml = ""

            try:

                candidate_plantuml = (
                    self.plantuml.render(
                        state.diagram
                    )
                )

                self.log.debug(
                    "PlantUML compilation "
                    "succeeded."
                )

            except Exception as exc:

                self.log.warning(
                    "PlantUML compiler failed: %s",
                    exc,
                )

                defects.append(
                    Defect(
                        id=(
                            f"COMPILER-{iteration}"
                        ),
                        category="SYNTAX",
                        severity=Severity.HIGH,
                        description=(
                            "Activity Diagram "
                            "could not be compiled "
                            "to PlantUML."
                        ),
                        node_ids=[],
                        edge_ids=[],
                        requirement_ids=[],
                        evidence=str(exc),
                        suggested_action=(
                            "Repair the Activity "
                            "Diagram structure."
                        ),
                    )
                )

            # ====================================================
            # 3. ACTUAL PLANTUML VALIDATION
            # ====================================================

            if candidate_plantuml:

                (
                    plantuml_ok,
                    plantuml_message,
                ) = (
                    self.plantuml_validator.validate(
                        candidate_plantuml
                    )
                )

                if not plantuml_ok:

                    self.log.warning(
                        "PlantUML validation failed."
                    )

                    defects.append(
                        Defect(
                            id=(
                                f"PLANTUML-"
                                f"{iteration}"
                            ),
                            category="SYNTAX",
                            severity=Severity.HIGH,
                            description=(
                                "Generated "
                                "PlantUML failed "
                                "compilation."
                            ),
                            node_ids=[],
                            edge_ids=[],
                            requirement_ids=[],
                            evidence=(
                                plantuml_message
                            ),
                            suggested_action=(
                                "Repair the Activity "
                                "Diagram so the "
                                "generated PlantUML "
                                "compiles."
                            ),
                        )
                    )

            # ====================================================
            # 4. SEMANTIC REVIEW
            # ====================================================

            self.log.info(
                "Running semantic reviewer..."
            )

            review = (
                self.reviewer.run(
                    requirement_text,
                    state.requirements,
                    state.requirement_matrix,
                    state.diagram,
                    existing_defects=defects,
                )
            )

            state.review = review

            defects.extend(
                review.defects
            )

            # ====================================================
            # 5. DEDUPLICATE
            # ====================================================

            defects = (
                self._deduplicate_defects(
                    defects
                )
            )

            state.defects = defects

            # ====================================================
            # 6. SCORE CANDIDATE
            # ====================================================

            score = (
                self._candidate_score(
                    validation,
                    review,
                    defects,
                )
            )

            self.log.info(
                "Candidate score: %.3f",
                score,
            )

            # ====================================================
            # 7. BEST CANDIDATE
            # ====================================================

            if (
                best_validation is None
                or score > best_score
            ):

                best_score = score

                best_iteration = (
                    iteration
                )

                best_diagram = (
                    state.diagram.model_copy(
                        deep=True
                    )
                )

                best_validation = (
                    validation.model_copy(
                        deep=True
                    )
                )

                best_review = (
                    review.model_copy(
                        deep=True
                    )
                )

                self.log.info(
                    "New best candidate "
                    "selected at iteration %d.",
                    iteration,
                )

            # ====================================================
            # 8. SUCCESS
            # ====================================================

            if not defects:

                self.log.info(
                    "Candidate passed all "
                    "current validations."
                )

                break

            # ====================================================
            # 9. ITERATION LIMIT
            # ====================================================

            if (
                iteration
                >= max_iterations
            ):

                self.log.warning(
                    "Maximum repair iterations "
                    "reached."
                )

                break

            # ====================================================
            # 10. FEEDBACK / PRIORITIZATION
            # ====================================================

            feedback = (
                self.feedback.run(
                    defects
                )
            )

            prioritized = (
                feedback.prioritized_defects
                or defects
            )

            # Keep repair batches intentionally small.
            prioritized = (
                self.repair_router.select_defects(
                    prioritized
                )
            )

            self.log.info(
                "Selected defects for repair: %s",
                [
                    defect.id
                    for defect in prioritized
                ],
            )

            # ====================================================
            # 11. PRESERVE CURRENT CANDIDATE
            # ====================================================

            original_diagram = (
                state.diagram.model_copy(
                    deep=True
                )
            )

            original_score = score

            # ====================================================
            # 12. REPAIR
            # ====================================================

            repair = (
                self.repair_router.repair(
                    requirement_text,
                    state.requirements,
                    state.diagram,
                    prioritized,
                )
            )

            if not repair.changed:

                self.log.warning(
                    "Repair agent made no changes. "
                    "Stopping repair loop."
                )

                break

            candidate_after_repair = (
                repair.diagram
            )

            # ====================================================
            # 13. IMMEDIATE STRUCTURAL REGRESSION
            # ====================================================

            regression_validation = (
                self.validator.validate(
                    candidate_after_repair,
                    state.requirements,
                )
            )

            regression_defects = list(
                regression_validation.defects
            )

            # ----------------------------------------------------
            # Semantic regression review
            # ----------------------------------------------------

            regression_review = (
                self.reviewer.run(
                    requirement_text,
                    state.requirements,
                    state.requirement_matrix,
                    candidate_after_repair,
                    existing_defects=(
                        regression_defects
                    ),
                )
            )

            regression_defects.extend(
                regression_review.defects
            )

            regression_defects = (
                self._deduplicate_defects(
                    regression_defects
                )
            )

            regression_score = (
                self._candidate_score(
                    regression_validation,
                    regression_review,
                    regression_defects,
                )
            )

            self.log.info(
                "Repaired candidate score: %.3f",
                regression_score,
            )

            # ====================================================
            # 14. ACCEPT OR ROLLBACK
            # ====================================================

            if (
                regression_score
                > original_score
            ):

                self.log.info(
                    "Repair improved the candidate "
                    "(%.3f -> %.3f).",
                    original_score,
                    regression_score,
                )

                state.diagram = (
                    candidate_after_repair
                )

                state.repair_history.append(
                    self._repair_record(
                        iteration=iteration,
                        repair=repair,
                        selected_defects=(
                            prioritized
                        ),
                        accepted=True,
                        before_score=(
                            original_score
                        ),
                        after_score=(
                            regression_score
                        ),
                    )
                )

            else:

                self.log.warning(
                    "Repair did not improve the "
                    "candidate (%.3f -> %.3f). "
                    "Rolling back.",
                    original_score,
                    regression_score,
                )

                state.diagram = (
                    original_diagram
                )

                state.repair_history.append(
                    self._repair_record(
                        iteration=iteration,
                        repair=repair,
                        selected_defects=(
                            prioritized
                        ),
                        accepted=False,
                        before_score=(
                            original_score
                        ),
                        after_score=(
                            regression_score
                        ),
                    )
                )

                # No reason to repeat the same unsuccessful
                # repair from the same state.
                break

        # ========================================================
        # 15. RESTORE BEST CANDIDATE
        # ========================================================

        if best_validation is not None:
            state.diagram = (
                best_diagram
            )

            state.validation = (
                best_validation
            )

            state.review = (
                best_review
            )

            state.defects = (
                self._deduplicate_defects(
                    (
                        best_validation.defects
                        if best_validation
                        else []
                    )
                    + (
                        best_review.defects
                        if best_review
                        else []
                    )
                )
            )

        # ========================================================
        # 16. FINAL PLANTUML
        # ========================================================

        self.log.info(
            "Compiling final diagram "
            "(best iteration: %d, "
            "score: %.3f)...",
            best_iteration,
            best_score,
        )

        try:

            state.final_plantuml = (
                self.plantuml.render(
                    state.diagram
                )
            )

        except Exception as exc:

            self.log.error(
                "Final PlantUML compilation failed: %s",
                exc,
            )

            state.final_plantuml = ""

            state.defects.append(
                Defect(
                    id="FINAL-COMPILER",
                    category="SYNTAX",
                    severity=Severity.CRITICAL,
                    description=(
                        "Final diagram could not "
                        "be compiled to PlantUML."
                    ),
                    node_ids=[],
                    edge_ids=[],
                    requirement_ids=[],
                    evidence=str(exc),
                    suggested_action=(
                        "Fix the final Activity "
                        "Diagram representation."
                    ),
                )
            )

        # ========================================================
        # 17. FINAL RENDERING
        # ========================================================

        output_path = Path(
            output_dir
        )

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        render_info: dict[str, Any] = {}

        if state.final_plantuml:

            try:

                render_info = (
                    self.renderer.render(
                        state.final_plantuml,
                        output_path,
                        name=sample_id,
                    )
                )

            except Exception as exc:

                self.log.warning(
                    "Rendering failed: %s",
                    exc,
                )

                render_info = {
                    "rendered": False,
                    "error": str(exc),
                }

        state.metrics["render"] = (
            render_info
        )

        state.metrics[
            "best_iteration"
        ] = best_iteration

        state.metrics[
            "best_score"
        ] = best_score

        state.metrics[
            "repair_iterations"
        ] = len(
            state.repair_history
        )

        state.metrics[
            "final_defect_count"
        ] = len(
            state.defects
        )

        state.metrics[
            "initial_requirement_count"
        ] = len(
            state.requirements
        )

        self.log.info(
            "Pipeline completed. "
            "Best iteration=%d, "
            "remaining defects=%d.",
            best_iteration,
            len(
                state.defects
            ),
        )

        return state

    # ============================================================
    # CANDIDATE SCORING
    # ============================================================

    @staticmethod
    def _candidate_score(
        validation: ValidationResult,
        review: ReviewResult,
        defects: list[Defect],
    ) -> float:

        structural_score = (
            validation.score
            if validation
            else 0.0
        )

        semantic_score = (
            review.semantic_score
            if review
            else 0.0
        )

        severity_penalty = 0.0

        severity_weights = {
            "CRITICAL": 15.0,
            "HIGH": 8.0,
            "MEDIUM": 3.0,
            "LOW": 1.0,
        }

        category_weights = {
            "SYNTAX": 2.0,
            "STRUCTURAL": 2.0,
            "DECISION": 2.0,
            "CONTROL_FLOW": 2.0,
            "CONCURRENCY": 2.0,
            "TERMINATION": 2.0,
            "REQUIREMENT_COVERAGE": 2.0,
            "SEMANTIC": 2.0,
            "EXCEPTION": 1.5,
            "HALLUCINATION": 1.5,
            "DATA_FLOW": 1.0,
            "GRANULARITY": 0.5,
            "LAYOUT": 0.25,
        }

        for defect in defects:

            severity_penalty += (
                severity_weights.get(
                    defect.severity.value,
                    1.0,
                )
                * category_weights.get(
                    defect.category,
                    1.0,
                )
            )

        return (
            (
                structural_score
                * 50.0
            )
            + (
                semantic_score
                * 50.0
            )
            - severity_penalty
        )

    # ============================================================
    # DEFECT DEDUPLICATION
    # ============================================================

    @staticmethod
    def _deduplicate_defects(
        defects: list[Defect],
    ) -> list[Defect]:

        result: list[Defect] = []

        seen: set[tuple] = set()

        for defect in defects:

            key = (
                defect.category,
                tuple(
                    sorted(
                        defect.node_ids
                    )
                ),
                tuple(
                    sorted(
                        defect.edge_ids
                    )
                ),
                tuple(
                    sorted(
                        defect.requirement_ids
                    )
                ),
                defect.description.strip().lower(),
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(
                defect
            )

        return result

    # ============================================================
    # REPAIR HISTORY
    # ============================================================

    @staticmethod
    def _repair_record(
        *,
        iteration: int,
        repair,
        selected_defects: list[Defect],
        accepted: bool,
        before_score: float,
        after_score: float,
    ) -> dict[str, Any]:

        return {
            "iteration": iteration,

            "repair_type": (
                repair.repair_type
            ),

            "changed": (
                repair.changed
            ),

            "accepted": accepted,

            "before_score": (
                before_score
            ),

            "after_score": (
                after_score
            ),

            "score_delta": (
                after_score
                - before_score
            ),

            "changes": (
                repair.changes
            ),

            "rationale": (
                repair.rationale
            ),

            "defect_ids": [
                defect.id
                for defect
                in selected_defects
            ],

            "defect_categories": [
                defect.category
                for defect
                in selected_defects
            ],
        }