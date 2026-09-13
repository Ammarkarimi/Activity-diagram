from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.agents.feedback_agent import FeedbackAgent
from src.agents.generator_agent import GeneratorAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.requirement_agent import RequirementAgent
from src.agents.reviewer_agent import ReviewerAgent

from src.generation.plantuml_generator import PlantUMLGenerator
from src.generation.ir_sanitizer import IRSanitizer
from src.generation.renderer import PlantUMLRenderer

from src.llm.openai_client import OpenAIClient

from src.models.domain import (
    ActivityDiagram,
    CandidateMetrics,
    CandidateRecord,
    Defect,
    HumanRating,
    PipelineState,
    Requirement,
    RequirementCoverage,
    RequirementMatrix,
    ReviewResult,
    Severity,
    ValidationResult,
)

from src.pipeline.repair_router import RepairRouter

from src.validation.syntax.plantuml_validator import (
    PlantUMLSyntaxValidator,
)

from src.validation.validator import HybridValidator


class MultiAgentPipeline:
    """
    Multi-agent requirement-to-activity-diagram pipeline.

    Main stages:

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
        Generator Agent
            |
            v
        Initial Candidate
            |
            +------------------------------+
            |                              |
            v                              v
      Deterministic Validation       Semantic Review
            |                              |
            +--------------+---------------+
                           |
                           v
                     Defect Set
                           |
                           v
                    Feedback Agent
                           |
                           v
                     Repair Router
                           |
                           v
                    Repair Candidate
                           |
                           v
                    Regression Check
                           |
                  +--------+--------+
                  |                 |
               Better            Worse
                  |                 |
                  v                 v
                Keep             Rollback
                  |
                  v
             Next Iteration
                  |
                  v
          Best Candidate Selection
                  |
                  v
           PlantUML Compilation
                  |
                  v
              Rendering

    Research-oriented tracking:

        - Candidate metrics for every iteration
        - Requirement coverage
        - Correctness
        - Completeness
        - Structural validity
        - Unsupported behaviour
        - Defect count
        - Defects fixed
        - Defects introduced
        - Defect reduction rate
        - Repair success rate
        - Candidate quality score
        - Execution time
        - LLM calls
        - Best candidate iteration
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

        self.requirement_agent = RequirementAgent(
            self.llm
        )

        self.planning_agent = PlanningAgent(
            self.llm
        )

        self.generator = GeneratorAgent(
            self.llm
        )

        self.reviewer = ReviewerAgent(
            self.llm
        )

        self.feedback = FeedbackAgent(
            self.llm
        )

        # --------------------------------------------------------
        # Pipeline components
        # --------------------------------------------------------

        self.repair_router = RepairRouter(
            self.llm,
            max_defects_per_repair=(
                max_defects_per_repair
            ),
        )

        self.validator = HybridValidator()

        self.plantuml = PlantUMLGenerator()

        self.plantuml_validator = (
            PlantUMLSyntaxValidator()
        )

        self.renderer = PlantUMLRenderer()

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

        pipeline_start = time.perf_counter()

        # ========================================================
        # INITIAL STATE
        # ========================================================

        state = PipelineState(
            sample_id=sample_id,
            requirement_text=requirement_text,
            requirements=[],
            requirement_matrix=RequirementMatrix(
                items=[]
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
            candidates=[],
            best_candidate_iteration=None,
            best_candidate=None,
        )

        # ========================================================
        # LLM CALL TRACKING
        # ========================================================

        llm_calls = 0

        def count_llm_call() -> None:
            nonlocal llm_calls
            llm_calls += 1

        # ========================================================
        # PHASE 1 — REQUIREMENT EXTRACTION
        # ========================================================

        self.log.info(
            "Running requirement agent..."
        )

        count_llm_call()

        extracted = self.requirement_agent.run(
            requirement_text
        )

        state.requirements = (
            extracted.requirements
        )

        state.requirement_matrix = (
            extracted.matrix
        )

        self.log.info(
            "Extracted %d requirements.",
            len(state.requirements),
        )

        # ========================================================
        # PHASE 2 — PLANNING
        # ========================================================

        self.log.info(
            "Running planning agent..."
        )

        count_llm_call()

        state.plan = self.planning_agent.run(
            requirement_text,
            state.requirements,
            state.requirement_matrix,
        )

        self.log.info(
            "Planning completed: "
            "%d nodes, %d edges, "
            "%d decisions, %d loops, "
            "%d concurrency blocks.",
            len(state.plan.nodes),
            len(state.plan.edges),
            len(state.plan.decisions),
            len(state.plan.loops),
            len(state.plan.concurrency),
        )

        # ========================================================
        # PHASE 3 — INITIAL GENERATION
        # ========================================================

        self.log.info(
            "Running generator agent..."
        )

        count_llm_call()

        state.diagram = self.generator.run(
            requirement_text,
            state.requirements,
            state.plan,
        )

        state.diagram = IRSanitizer.sanitize(
            state.diagram
        )

        self.log.info(
            "Generated diagram: "
            "%d nodes, %d edges.",
            len(state.diagram.nodes),
            len(state.diagram.edges),
        )

        # ========================================================
        # BEST CANDIDATE TRACKING
        # ========================================================

        best_diagram = state.diagram.model_copy(
            deep=True
        )

        best_validation: ValidationResult | None = None

        best_review: ReviewResult | None = None

        best_score = float("-inf")

        best_iteration = 0

        # ========================================================
        # INITIAL DEFECT COUNT
        # ========================================================

        previous_defect_ids: set[str] = set()

        previous_defect_count = 0

        # ========================================================
        # ITERATIVE VALIDATION / REPAIR
        # ========================================================

        for iteration in range(
            max_iterations + 1
        ):

            state.iteration = iteration

            iteration_start = time.perf_counter()

            self.log.info(
                "Validation iteration %d",
                iteration,
            )

            # ====================================================
            # 1. DETERMINISTIC VALIDATION
            # ====================================================

            validation = self.validator.validate(
                state.diagram,
                state.requirements,
            )

            state.validation = validation

            defects: list[Defect] = list(
                validation.defects
            )

            self.log.info(
                "Deterministic validation "
                "found %d defect(s).",
                len(validation.defects),
            )

            # ====================================================
            # 2. PLANTUML COMPILATION
            # ====================================================

            candidate_plantuml = ""

            try:

                candidate_plantuml = (
                    self.plantuml.render(
                        state.diagram
                    )
                )

                self.log.debug(
                    "PlantUML compilation succeeded."
                )

            except Exception as exc:

                self.log.warning(
                    "PlantUML compiler failed: %s",
                    exc,
                )

                defects.append(
                    Defect(
                        id=f"COMPILER-{iteration}",
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
            # 3. PLANTUML VALIDATION
            # ====================================================

            if candidate_plantuml:

                (
                    plantuml_ok,
                    plantuml_message,
                ) = self.plantuml_validator.validate(
                    candidate_plantuml
                )

                if not plantuml_ok:

                    self.log.warning(
                        "PlantUML validation failed."
                    )

                    defects.append(
                        Defect(
                            id=f"PLANTUML-{iteration}",
                            category="SYNTAX",
                            severity=Severity.HIGH,
                            description=(
                                "Generated PlantUML "
                                "failed compilation."
                            ),
                            node_ids=[],
                            edge_ids=[],
                            requirement_ids=[],
                            evidence=plantuml_message,
                            suggested_action=(
                                "Repair the Activity "
                                "Diagram so generated "
                                "PlantUML compiles."
                            ),
                        )
                    )

            # ====================================================
            # 4. SEMANTIC REVIEW
            # ====================================================

            self.log.info(
                "Running semantic reviewer..."
            )

            count_llm_call()

            review = self.reviewer.run(
                requirement_text,
                state.requirements,
                state.requirement_matrix,
                state.diagram,
                existing_defects=defects,
            )

            state.review = review

            defects.extend(
                review.defects
            )

            # ====================================================
            # 5. DEDUPLICATION
            # ====================================================

            defects = self._deduplicate_defects(
                defects
            )

            state.defects = defects

            # ====================================================
            # 6. REQUIREMENT COVERAGE
            # ====================================================

            requirement_coverage = (
                self._calculate_requirement_coverage(
                    state.requirements,
                    state.diagram,
                )
            )

            coverage_score = (
                self._coverage_score(
                    requirement_coverage
                )
            )

            # ====================================================
            # 7. CORRECTNESS
            # ====================================================

            correctness = (
                self._calculate_correctness(
                    validation=validation,
                    review=review,
                    defects=defects,
                )
            )

            # ====================================================
            # 8. COMPLETENESS
            # ====================================================

            completeness = (
                self._calculate_completeness(
                    state.requirements,
                    requirement_coverage,
                )
            )

            # ====================================================
            # 9. STRUCTURAL VALIDITY
            # ====================================================

            structural_validity = (
                self._calculate_structural_validity(
                    validation
                )
            )

            # ====================================================
            # 10. UNSUPPORTED BEHAVIOUR
            # ====================================================

            unsupported_behaviour_rate = (
                self._calculate_unsupported_behaviour_rate(
                    review,
                    state.diagram,
                )
            )

            # ====================================================
            # 11. DEFECT METRICS
            # ====================================================

            current_defect_ids = {
                defect.id
                for defect in defects
            }

            current_defect_count = len(
                defects
            )

            defects_fixed = len(
                previous_defect_ids
                - current_defect_ids
            )

            defects_introduced = len(
                current_defect_ids
                - previous_defect_ids
            )

            if previous_defect_count > 0:

                defect_reduction_rate = (
                    previous_defect_count
                    - current_defect_count
                ) / previous_defect_count

            else:

                defect_reduction_rate = 0.0

            defect_reduction_rate = max(
                0.0,
                min(
                    1.0,
                    defect_reduction_rate,
                ),
            )

            # ====================================================
            # 12. CANDIDATE QUALITY SCORE
            # ====================================================

            score = self._candidate_score(
                validation,
                review,
                defects,
            )

            quality_score = (
                self._quality_score(
                    requirement_coverage=coverage_score,
                    correctness=correctness,
                    completeness=completeness,
                    structural_validity=(
                        structural_validity
                    ),
                    unsupported_behaviour_rate=(
                        unsupported_behaviour_rate
                    ),
                    defect_count=current_defect_count,
                    candidate_score=score,
                )
            )

            # ====================================================
            # 13. ITERATION TIME
            # ====================================================

            iteration_time = (
                time.perf_counter()
                - iteration_start
            )

            # ====================================================
            # 14. CANDIDATE METRICS
            # ====================================================

            candidate_metrics = CandidateMetrics(
                iteration=iteration,

                requirement_coverage=(
                    coverage_score
                ),

                correctness=(
                    correctness
                ),

                completeness=(
                    completeness
                ),

                structural_validity=(
                    structural_validity
                ),

                unsupported_behaviour_rate=(
                    unsupported_behaviour_rate
                ),

                defect_count=(
                    current_defect_count
                ),

                defects_fixed=(
                    defects_fixed
                ),

                defects_introduced=(
                    defects_introduced
                ),

                defect_reduction_rate=(
                    defect_reduction_rate
                ),

                repair_success_rate=0.0,

                quality_score=(
                    quality_score
                ),

                execution_time_seconds=(
                    iteration_time
                ),

                llm_calls=llm_calls,
            )

            # ====================================================
            # 15. STORE CANDIDATE
            # ====================================================

            candidate_record = CandidateRecord(
                iteration=iteration,

                diagram=state.diagram.model_copy(
                    deep=True
                ),

                plantuml=candidate_plantuml,

                metrics=candidate_metrics,

                defects=[
                    defect.model_copy(
                        deep=True
                    )
                    for defect in defects
                ],

                requirement_coverage=(
                    requirement_coverage
                ),
            )

            state.candidates.append(
                candidate_record
            )

            # ====================================================
            # 16. LOG METRICS
            # ====================================================

            self.log.info(
                "Iteration %d metrics: "
                "coverage=%.3f, "
                "correctness=%.3f, "
                "completeness=%.3f, "
                "structural=%.3f, "
                "unsupported=%.3f, "
                "defects=%d, "
                "quality=%.3f",
                iteration,
                coverage_score,
                correctness,
                completeness,
                structural_validity,
                unsupported_behaviour_rate,
                current_defect_count,
                quality_score,
            )

            # ====================================================
            # 17. BEST CANDIDATE
            # ====================================================

            if (
                best_validation is None
                or score > best_score
            ):

                best_score = score

                best_iteration = iteration

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
                    "New best candidate selected "
                    "at iteration %d "
                    "(score=%.3f).",
                    iteration,
                    score,
                )

            # ====================================================
            # 18. SUCCESS
            # ====================================================

            if not defects:

                self.log.info(
                    "Candidate passed all "
                    "current validations."
                )

                break

            # ====================================================
            # 19. ITERATION LIMIT
            # ====================================================

            if iteration >= max_iterations:

                self.log.warning(
                    "Maximum repair iterations reached."
                )

                break

            # ====================================================
            # 20. FEEDBACK
            # ====================================================

            feedback = self.feedback.run(
                defects
            )

            count_llm_call()

            prioritized = (
                feedback.prioritized_defects
                or defects
            )

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
            # 21. PRESERVE CURRENT CANDIDATE
            # ====================================================

            original_diagram = (
                state.diagram.model_copy(
                    deep=True
                )
            )

            original_score = score

            # ====================================================
            # 22. REPAIR
            # ====================================================

            repair = self.repair_router.repair(
                requirement_text,
                state.requirements,
                state.diagram,
                prioritized,
            )

            count_llm_call()

            if not repair.changed:

                self.log.warning(
                    "Repair agent made no changes. "
                    "Stopping repair loop."
                )

                break

            candidate_after_repair = (
                IRSanitizer.sanitize(
                    repair.diagram
                )
            )

            # ====================================================
            # 23. REGRESSION VALIDATION
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

            # ====================================================
            # 24. REGRESSION SEMANTIC REVIEW
            # ====================================================

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

            count_llm_call()

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
            # 25. REPAIR SUCCESS
            # ====================================================

            repair_success = (
                regression_score
                > original_score
            )

            # Update the candidate record's repair metric.

            if state.candidates:

                state.candidates[-1].metrics.repair_success_rate = (
                    1.0
                    if repair_success
                    else 0.0
                )

            # ====================================================
            # 26. ACCEPT / ROLLBACK
            # ====================================================

            if repair_success:

                self.log.info(
                    "Repair improved candidate "
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
                        selected_defects=prioritized,
                        accepted=True,
                        before_score=original_score,
                        after_score=regression_score,
                    )
                )

            else:

                self.log.warning(
                    "Repair did not improve "
                    "candidate "
                    "(%.3f -> %.3f). "
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
                        selected_defects=prioritized,
                        accepted=False,
                        before_score=original_score,
                        after_score=regression_score,
                    )
                )

                break

            # ====================================================
            # 27. UPDATE PREVIOUS DEFECT STATE
            # ====================================================

            previous_defect_ids = (
                current_defect_ids
            )

            previous_defect_count = (
                current_defect_count
            )

        # ========================================================
        # RESTORE BEST CANDIDATE
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

            state.best_candidate_iteration = (
                best_iteration
            )

            # Locate the actual best candidate record.

            for candidate in state.candidates:

                if (
                    candidate.iteration
                    == best_iteration
                ):

                    state.best_candidate = (
                        candidate
                    )

                    break

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
        # FINAL PLANTUML
        # ========================================================

        self.log.info(
            "Compiling final diagram "
            "(best iteration: %d, score: %.3f)...",
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
        # FINAL RENDERING
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

        # ========================================================
        # FINAL METRICS
        # ========================================================

        total_execution_time = (
            time.perf_counter()
            - pipeline_start
        )

        state.metrics["render"] = (
            render_info
        )

        state.metrics["best_iteration"] = (
            best_iteration
        )

        state.metrics["best_score"] = (
            best_score
        )

        state.metrics["best_candidate_quality"] = (
            (
                state.best_candidate.metrics.quality_score
                if state.best_candidate
                else 0.0
            )
        )

        state.metrics["repair_iterations"] = (
            len(state.repair_history)
        )

        state.metrics["final_defect_count"] = (
            len(state.defects)
        )

        state.metrics["initial_requirement_count"] = (
            len(state.requirements)
        )

        state.metrics["candidate_count"] = (
            len(state.candidates)
        )

        state.metrics["total_execution_time_seconds"] = (
            total_execution_time
        )

        state.metrics["llm_calls"] = (
            llm_calls
        )

        state.metrics["candidate_scores"] = [
            {
                "iteration": candidate.iteration,
                "quality_score": (
                    candidate.metrics.quality_score
                ),
                "defect_count": (
                    candidate.metrics.defect_count
                ),
                "requirement_coverage": (
                    candidate.metrics.requirement_coverage
                ),
                "correctness": (
                    candidate.metrics.correctness
                ),
                "completeness": (
                    candidate.metrics.completeness
                ),
                "structural_validity": (
                    candidate.metrics.structural_validity
                ),
                "unsupported_behaviour_rate": (
                    candidate.metrics.unsupported_behaviour_rate
                ),
            }
            for candidate in state.candidates
        ]

        # ========================================================
        # RESEARCH SUMMARY
        # ========================================================

        state.metrics["research_summary"] = (
            self._build_research_summary(
                state
            )
        )

        self.log.info(
            "Pipeline completed. "
            "Best iteration=%d, "
            "remaining defects=%d, "
            "quality=%.3f.",
            best_iteration,
            len(state.defects),
            state.metrics[
                "best_candidate_quality"
            ],
        )

        return state

    # ============================================================
    # CANDIDATE SCORE
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

        severity_penalty = 0.0

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
    # QUALITY SCORE
    # ============================================================

    @staticmethod
    def _quality_score(
        requirement_coverage: float,
        correctness: float,
        completeness: float,
        structural_validity: float,
        unsupported_behaviour_rate: float,
        defect_count: int,
        candidate_score: float,
    ) -> float:
        """
        Research-oriented normalized quality score.

        Higher is better.

        Components:

            Requirement Coverage   25%
            Correctness            25%
            Completeness           20%
            Structural Validity    15%
            Unsupported Behaviour  10%
            Defect-free factor      5%
        """

        unsupported_score = (
            1.0
            - max(
                0.0,
                min(
                    1.0,
                    unsupported_behaviour_rate,
                ),
            )
        )

        defect_free_score = (
            1.0
            if defect_count == 0
            else 1.0
            / (
                1.0
                + defect_count
            )
        )

        score = (
            0.25 * requirement_coverage
            + 0.25 * correctness
            + 0.20 * completeness
            + 0.15 * structural_validity
            + 0.10 * unsupported_score
            + 0.05 * defect_free_score
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # ============================================================
    # REQUIREMENT COVERAGE
    # ============================================================

    @staticmethod
    def _calculate_requirement_coverage(
        requirements: list[Requirement],
        diagram: ActivityDiagram,
    ) -> list[RequirementCoverage]:

        result: list[RequirementCoverage] = []

        for requirement in requirements:

            node_matches = [
                node
                for node in diagram.nodes
                if requirement.id
                in node.requirement_ids
            ]

            edge_matches = [
                edge
                for edge in diagram.edges
                if requirement.id
                in edge.requirement_ids
            ]

            covered = bool(
                node_matches
                or edge_matches
            )

            if (
                node_matches
                and edge_matches
            ):

                coverage_type = "FULL"

            elif node_matches:

                coverage_type = "NODE"

            elif edge_matches:

                coverage_type = "EDGE"

            else:

                coverage_type = "NONE"

            evidence_parts = []

            if node_matches:

                evidence_parts.append(
                    "Nodes: "
                    + ", ".join(
                        node.id
                        for node in node_matches
                    )
                )

            if edge_matches:

                evidence_parts.append(
                    "Edges: "
                    + ", ".join(
                        edge.id
                        for edge in edge_matches
                    )
                )

            result.append(
                RequirementCoverage(
                    requirement_id=(
                        requirement.id
                    ),
                    covered=covered,
                    coverage_type=(
                        coverage_type
                    ),
                    evidence="; ".join(
                        evidence_parts
                    ),
                )
            )

        return result

    # ============================================================
    # COVERAGE SCORE
    # ============================================================

    @staticmethod
    def _coverage_score(
        coverage: list[RequirementCoverage],
    ) -> float:

        if not coverage:
            return 1.0

        covered = sum(
            1
            for item in coverage
            if item.covered
        )

        return (
            covered
            / len(coverage)
        )

    # ============================================================
    # CORRECTNESS
    # ============================================================

    @staticmethod
    def _calculate_correctness(
        validation: ValidationResult,
        review: ReviewResult,
        defects: list[Defect],
    ) -> float:

        structural = (
            validation.score
        )

        semantic = (
            review.semantic_score
        )

        critical_or_high = sum(
            1
            for defect in defects
            if defect.severity
            in {
                Severity.CRITICAL,
                Severity.HIGH,
            }
        )

        major_defect_penalty = min(
            1.0,
            critical_or_high * 0.15,
        )

        correctness = (
            (
                0.5
                * structural
            )
            + (
                0.5
                * semantic
            )
            - major_defect_penalty
        )

        return max(
            0.0,
            min(
                1.0,
                correctness,
            ),
        )

    # ============================================================
    # COMPLETENESS
    # ============================================================

    @staticmethod
    def _calculate_completeness(
        requirements: list[Requirement],
        coverage: list[RequirementCoverage],
    ) -> float:

        if not requirements:
            return 1.0

        covered = sum(
            1
            for item in coverage
            if item.covered
        )

        return (
            covered
            / len(requirements)
        )

    # ============================================================
    # STRUCTURAL VALIDITY
    # ============================================================

    @staticmethod
    def _calculate_structural_validity(
        validation: ValidationResult,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                validation.score,
            ),
        )

    # ============================================================
    # UNSUPPORTED BEHAVIOUR RATE
    # ============================================================

    @staticmethod
    def _calculate_unsupported_behaviour_rate(
        review: ReviewResult,
        diagram: ActivityDiagram,
    ) -> float:

        node_count = max(
            1,
            len(diagram.nodes),
        )

        unsupported_count = len(
            review.unsupported_behaviors
        )

        return max(
            0.0,
            min(
                1.0,
                unsupported_count
                / node_count,
            ),
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

                defect.description
                .strip()
                .lower(),
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

            "accepted": (
                accepted
            ),

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
                for defect in selected_defects
            ],

            "defect_categories": [
                defect.category
                for defect in selected_defects
            ],
        }

    # ============================================================
    # RESEARCH SUMMARY
    # ============================================================

    @staticmethod
    def _build_research_summary(
        state: PipelineState,
    ) -> dict[str, Any]:

        if not state.candidates:

            return {
                "candidate_count": 0,
                "best_iteration": None,
                "best_quality": 0.0,
                "initial_defects": 0,
                "final_defects": len(
                    state.defects
                ),
            }

        first = state.candidates[0]

        best = max(
            state.candidates,
            key=lambda candidate: (
                candidate.metrics.quality_score
            ),
        )

        final = state.candidates[-1]

        initial_defects = (
            first.metrics.defect_count
        )

        final_defects = (
            final.metrics.defect_count
        )

        if initial_defects > 0:

            overall_defect_reduction = (
                initial_defects
                - final_defects
            ) / initial_defects

        else:

            overall_defect_reduction = 0.0

        return {
            "candidate_count": len(
                state.candidates
            ),

            "initial_iteration": (
                first.iteration
            ),

            "best_iteration": (
                best.iteration
            ),

            "best_quality_score": (
                best.metrics.quality_score
            ),

            "final_quality_score": (
                final.metrics.quality_score
            ),

            "initial_defects": (
                initial_defects
            ),

            "final_defects": (
                final_defects
            ),

            "overall_defect_reduction_rate": (
                overall_defect_reduction
            ),

            "best_requirement_coverage": (
                best.metrics.requirement_coverage
            ),

            "best_correctness": (
                best.metrics.correctness
            ),

            "best_completeness": (
                best.metrics.completeness
            ),

            "best_structural_validity": (
                best.metrics.structural_validity
            ),

            "best_unsupported_behaviour_rate": (
                best.metrics.unsupported_behaviour_rate
            ),

            "accepted_repairs": sum(
                1
                for repair
                in state.repair_history
                if repair.get(
                    "accepted",
                    False,
                )
            ),

            "rejected_repairs": sum(
                1
                for repair
                in state.repair_history
                if not repair.get(
                    "accepted",
                    False,
                )
            ),
        }