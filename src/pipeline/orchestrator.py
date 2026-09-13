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
from src.generation.ir_sanitizer import IRSanitizer
from src.generation.plantuml_generator import PlantUMLGenerator
from src.generation.renderer import PlantUMLRenderer
from src.llm.openai_client import OpenAIClient
from src.models.domain import (
    ActivityDiagram,
    CandidateMetrics,
    CandidateRecord,
    Defect,
    PipelineState,
    Requirement,
    RequirementCoverage,
    RequirementMatrix,
    ReviewResult,
    Severity,
    ValidationResult,
)
from src.pipeline.defect_utils import defect_key, defect_signatures
from src.pipeline.repair_router import RepairRouter
from src.pipeline.scoring import candidate_score, quality_score
from src.validation.syntax.plantuml_validator import PlantUMLSyntaxValidator
from src.validation.validator import HybridValidator


class MultiAgentPipeline:
    """Orchestrates the complete requirement-to-activity-diagram workflow.

    The orchestrator is the state manager. Agents do not communicate directly
    with one another: every output returns to this class, which decides what
    happens next and stores the candidate/repair history.

    Flow:
        Requirement -> Requirement Agent -> Planning Agent -> Generator Agent
        -> Validation + Semantic Review -> Feedback -> Repair Router
        -> Re-evaluation -> Accept/Rollback -> next iteration
        -> Best Candidate -> PlantUML -> Rendering
    """

    def __init__(self, model: str | None = None, max_defects_per_repair: int = 2) -> None:
        self.llm = OpenAIClient(model=model)

        self.requirement_agent = RequirementAgent(self.llm)
        self.planning_agent = PlanningAgent(self.llm)
        self.generator = GeneratorAgent(self.llm)
        self.reviewer = ReviewerAgent(self.llm)
        self.feedback = FeedbackAgent(self.llm)

        self.repair_router = RepairRouter(
            self.llm,
            max_defects_per_repair=max_defects_per_repair,
        )

        self.validator = HybridValidator()
        self.plantuml = PlantUMLGenerator()
        self.plantuml_validator = PlantUMLSyntaxValidator()
        self.renderer = PlantUMLRenderer()
        self.log = logging.getLogger(self.__class__.__name__)

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
        max_iterations = max(0, max_iterations)
        llm_calls = 0

        state = PipelineState(
            sample_id=sample_id,
            requirement_text=requirement_text,
            requirements=[],
            requirement_matrix=RequirementMatrix(items=[]),
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

        def call_llm() -> None:
            nonlocal llm_calls
            llm_calls += 1

        # --------------------------------------------------------
        # 1. Requirement extraction
        # --------------------------------------------------------
        self.log.info("Running requirement agent...")
        call_llm()
        extracted = self.requirement_agent.run(requirement_text)
        state.requirements = extracted.requirements
        state.requirement_matrix = extracted.matrix
        self.log.info("Extracted %d requirements.", len(state.requirements))

        # --------------------------------------------------------
        # 2. Planning
        # --------------------------------------------------------
        self.log.info("Running planning agent...")
        call_llm()
        state.plan = self.planning_agent.run(
            requirement_text,
            state.requirements,
            state.requirement_matrix,
        )
        self.log.info(
            "Planning completed: %d nodes, %d edges, %d decisions, %d loops, %d concurrency blocks.",
            len(state.plan.nodes),
            len(state.plan.edges),
            len(state.plan.decisions),
            len(state.plan.loops),
            len(state.plan.concurrency),
        )

        # --------------------------------------------------------
        # 3. Initial generation
        # --------------------------------------------------------
        self.log.info("Running generator agent...")
        call_llm()
        state.diagram = IRSanitizer.sanitize(
            self.generator.run(
                requirement_text,
                state.requirements,
                state.plan,
            )
        )
        self.log.info(
            "Generated diagram: %d nodes, %d edges.",
            len(state.diagram.nodes),
            len(state.diagram.edges),
        )

        best_quality = -1.0
        best_iteration = 0
        best_candidate: CandidateRecord | None = None
        previous_signatures: set[str] | None = None
        previous_defect_count: int | None = None

        # --------------------------------------------------------
        # 4. Iterative evaluation / repair loop
        # --------------------------------------------------------
        for iteration in range(max_iterations + 1):
            state.iteration = iteration
            iteration_start = time.perf_counter()
            self.log.info("Validation iteration %d", iteration)

            validation, review, defects, plantuml_text = self._evaluate_candidate(
                requirement_text=requirement_text,
                requirements=state.requirements,
                matrix=state.requirement_matrix,
                diagram=state.diagram,
                count_llm_call=call_llm,
            )

            state.validation = validation
            state.review = review
            state.defects = defects

            coverage = self._calculate_requirement_coverage(
                state.requirements, state.diagram
            )
            coverage_score = self._coverage_score(coverage)
            correctness = self._calculate_correctness(validation, review, defects)
            completeness = self._calculate_completeness(state.requirements, coverage)
            structural_validity = self._calculate_structural_validity(validation)
            unsupported_rate = self._calculate_unsupported_behaviour_rate(
                review, state.diagram
            )

            current_signatures = defect_signatures(defects)
            current_defect_count = len(defects)

            if previous_signatures is None:
                defects_fixed = 0
                defects_introduced = 0
                reduction_rate = 0.0
            else:
                defects_fixed = len(previous_signatures - current_signatures)
                defects_introduced = len(current_signatures - previous_signatures)
                reduction_rate = self._reduction_rate(
                    previous_defect_count or 0,
                    current_defect_count,
                )

            quality = quality_score(
                requirement_coverage=coverage_score,
                correctness=correctness,
                completeness=completeness,
                structural_validity=structural_validity,
                unsupported_behaviour_rate=unsupported_rate,
                defect_count=current_defect_count,
            )
            raw_score = candidate_score(validation, review, defects)

            iteration_time = time.perf_counter() - iteration_start
            metrics = CandidateMetrics(
                iteration=iteration,
                requirement_coverage=coverage_score,
                correctness=correctness,
                completeness=completeness,
                structural_validity=structural_validity,
                unsupported_behaviour_rate=unsupported_rate,
                defect_count=current_defect_count,
                defects_fixed=defects_fixed,
                defects_introduced=defects_introduced,
                defect_reduction_rate=reduction_rate,
                repair_success_rate=0.0,
                quality_score=quality,
                execution_time_seconds=iteration_time,
                llm_calls=llm_calls,
            )

            record = CandidateRecord(
                iteration=iteration,
                diagram=state.diagram.model_copy(deep=True),
                plantuml=plantuml_text,
                metrics=metrics,
                validation=validation.model_copy(deep=True),
                review=review.model_copy(deep=True),
                defects=[d.model_copy(deep=True) for d in defects],
                requirement_coverage=coverage,
            )
            state.candidates.append(record)

            self.log.info(
                "Iteration %d metrics: coverage=%.3f, correctness=%.3f, completeness=%.3f, structural=%.3f, unsupported=%.3f, defects=%d, quality=%.3f",
                iteration,
                coverage_score,
                correctness,
                completeness,
                structural_validity,
                unsupported_rate,
                current_defect_count,
                quality,
            )

            # Best candidate is selected using the same normalized quality
            # score that is reported to the research evaluation.
            if self._is_better_candidate(record, best_candidate):
                best_candidate = record.model_copy(deep=True)
                best_quality = quality
                best_iteration = iteration
                self.log.info(
                    "New best candidate selected at iteration %d (quality=%.3f, raw_score=%.3f).",
                    iteration,
                    quality,
                    raw_score,
                )

            # Update comparison baseline for the next evaluated candidate.
            previous_signatures = current_signatures
            previous_defect_count = current_defect_count

            if not defects:
                self.log.info("Candidate passed all current validations.")
                break

            if iteration >= max_iterations:
                self.log.warning("Maximum repair iterations reached.")
                break

            # ----------------------------------------------------
            # 5. Feedback / defect prioritization
            # ----------------------------------------------------
            self.log.info("Running feedback agent...")
            call_llm()
            feedback_result = self.feedback.run(defects)
            prioritized = self._filter_feedback_defects(
                feedback_result.prioritized_defects,
                defects,
            )
            prioritized = self.repair_router.select_defects(
                prioritized or defects
            )

            if not prioritized:
                self.log.warning("No valid repair defects were selected. Stopping.")
                break

            self.log.info(
                "Selected defects for repair: %s",
                [defect.id for defect in prioritized],
            )

            # ----------------------------------------------------
            # 6. Repair candidate
            # ----------------------------------------------------
            original_diagram = state.diagram.model_copy(deep=True)
            original_quality = quality
            original_defect_signatures = current_signatures

            uses_llm_repair = not self._is_deterministic_repair_batch(prioritized)
            repair = self.repair_router.repair(
                requirement_text,
                state.requirements,
                state.diagram,
                prioritized,
            )
            if uses_llm_repair:
                call_llm()

            if not repair.changed:
                self.log.warning("Repair made no changes. Stopping repair loop.")
                state.repair_history.append(
                    self._repair_record(
                        iteration=iteration,
                        repair=repair,
                        selected_defects=prioritized,
                        accepted=False,
                        before_score=original_quality,
                        after_score=original_quality,
                        before_defect_count=current_defect_count,
                        after_defect_count=current_defect_count,
                        defects_fixed=0,
                        defects_introduced=0,
                        reason="NO_CHANGE",
                    )
                )
                break

            repaired_diagram = IRSanitizer.sanitize(repair.diagram)

            # ----------------------------------------------------
            # 7. Regression evaluation
            # ----------------------------------------------------
            (
                regression_validation,
                regression_review,
                regression_defects,
                regression_plantuml,
            ) = self._evaluate_candidate(
                requirement_text=requirement_text,
                requirements=state.requirements,
                matrix=state.requirement_matrix,
                diagram=repaired_diagram,
                count_llm_call=call_llm,
            )

            regression_coverage = self._calculate_requirement_coverage(
                state.requirements, repaired_diagram
            )
            regression_coverage_score = self._coverage_score(regression_coverage)
            regression_correctness = self._calculate_correctness(
                regression_validation,
                regression_review,
                regression_defects,
            )
            regression_completeness = self._calculate_completeness(
                state.requirements,
                regression_coverage,
            )
            regression_structural = self._calculate_structural_validity(
                regression_validation
            )
            regression_unsupported = self._calculate_unsupported_behaviour_rate(
                regression_review,
                repaired_diagram,
            )
            regression_quality = quality_score(
                requirement_coverage=regression_coverage_score,
                correctness=regression_correctness,
                completeness=regression_completeness,
                structural_validity=regression_structural,
                unsupported_behaviour_rate=regression_unsupported,
                defect_count=len(regression_defects),
            )
            regression_signatures = defect_signatures(regression_defects)
            fixed = len(original_defect_signatures - regression_signatures)
            introduced = len(regression_signatures - original_defect_signatures)

            self.log.info(
                "Repair evaluation: quality %.3f -> %.3f, defects %d -> %d, fixed=%d, introduced=%d.",
                original_quality,
                regression_quality,
                current_defect_count,
                len(regression_defects),
                fixed,
                introduced,
            )

            # ----------------------------------------------------
            # 8. Accept only a non-regressing repair
            # ----------------------------------------------------
            accepted, reason = self._should_accept_repair(
                original_quality=original_quality,
                repaired_quality=regression_quality,
                original_defect_count=current_defect_count,
                repaired_defect_count=len(regression_defects),
                defects_fixed=fixed,
                defects_introduced=introduced,
            )

            state.repair_history.append(
                self._repair_record(
                    iteration=iteration,
                    repair=repair,
                    selected_defects=prioritized,
                    accepted=accepted,
                    before_score=original_quality,
                    after_score=regression_quality,
                    before_defect_count=current_defect_count,
                    after_defect_count=len(regression_defects),
                    defects_fixed=fixed,
                    defects_introduced=introduced,
                    reason=reason,
                )
            )

            if accepted:
                state.diagram = repaired_diagram
                self.log.info(
                    "Repair accepted: %s",
                    reason,
                )
                # The repaired candidate will be evaluated again in the next
                # iteration. This keeps candidate records comparable: every
                # stored candidate has passed through the same evaluation path.
            else:
                state.diagram = original_diagram
                self.log.warning("Repair rejected: %s", reason)
                if iteration >= max_iterations:
                    self.log.warning("Maximum repair iterations reached after rejected repair.")
                    break
                self.log.info(
                    "Continuing with the previous candidate to attempt another repair."
                )

        # --------------------------------------------------------
        # 9. Restore the best candidate
        # --------------------------------------------------------
        if best_candidate is not None:
            state.diagram = best_candidate.diagram.model_copy(deep=True)
            state.best_candidate = best_candidate.model_copy(deep=True)
            state.best_candidate_iteration = best_iteration
            state.validation = (
                best_candidate.validation.model_copy(deep=True)
                if best_candidate.validation is not None
                else None
            )
            state.review = (
                best_candidate.review.model_copy(deep=True)
                if best_candidate.review is not None
                else None
            )
            state.defects = [
                defect.model_copy(deep=True)
                for defect in best_candidate.defects
            ]

        # --------------------------------------------------------
        # 10. Final PlantUML with explicit non-empty check
        # --------------------------------------------------------
        state.final_plantuml, final_puml_defect = self._compile_plantuml(
            state.diagram,
            iteration=best_iteration,
        )
        if final_puml_defect is not None:
            state.defects = self._deduplicate_defects(
                state.defects + [final_puml_defect]
            )

        if not state.final_plantuml:
            state.final_plantuml = self._fallback_plantuml(sample_id, state)

        # --------------------------------------------------------
        # 11. Render
        # --------------------------------------------------------
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        render_info: dict[str, Any] = {}

        if state.final_plantuml:
            try:
                render_info = self.renderer.render(
                    state.final_plantuml,
                    output_path,
                    name=sample_id,
                )
            except Exception as exc:
                render_info = {
                    "rendered": False,
                    "error": str(exc),
                }
                self.log.warning("Rendering failed: %s", exc)
        else:
            render_info = {
                "rendered": False,
                "error": "Final PlantUML is empty.",
            }

        # --------------------------------------------------------
        # 12. Research metrics
        # --------------------------------------------------------
        total_execution_time = time.perf_counter() - pipeline_start
        state.metrics.update(
            {
                "render": render_info,
                "best_iteration": best_iteration,
                "best_score": best_quality * 100.0,
                "best_candidate_quality": best_quality,
                "repair_iterations": len(state.repair_history),
                "final_defect_count": len(state.defects),
                "initial_requirement_count": len(state.requirements),
                "candidate_count": len(state.candidates),
                "total_execution_time_seconds": total_execution_time,
                "llm_calls": llm_calls,
                "candidate_scores": [
                    {
                        "iteration": c.iteration,
                        "quality_score": c.metrics.quality_score,
                        "quality_score_100": c.metrics.quality_score * 100.0,
                        "defect_count": c.metrics.defect_count,
                        "defects_fixed": c.metrics.defects_fixed,
                        "defects_introduced": c.metrics.defects_introduced,
                        "defect_reduction_rate": c.metrics.defect_reduction_rate,
                        "requirement_coverage": c.metrics.requirement_coverage,
                        "correctness": c.metrics.correctness,
                        "completeness": c.metrics.completeness,
                        "structural_validity": c.metrics.structural_validity,
                        "unsupported_behaviour_rate": c.metrics.unsupported_behaviour_rate,
                        "execution_time_seconds": c.metrics.execution_time_seconds,
                        "llm_calls": c.metrics.llm_calls,
                    }
                    for c in state.candidates
                ],
            }
        )
        state.metrics["research_summary"] = self._build_research_summary(state)

        self.log.info(
            "Pipeline completed. Best iteration=%d, remaining defects=%d, quality=%.3f.",
            best_iteration,
            len(state.defects),
            best_quality,
        )
        return state

    # ============================================================
    # CANDIDATE EVALUATION
    # ============================================================

    def _evaluate_candidate(
        self,
        *,
        requirement_text: str,
        requirements: list[Requirement],
        matrix: RequirementMatrix,
        diagram: ActivityDiagram,
        count_llm_call,
    ) -> tuple[ValidationResult, ReviewResult, list[Defect], str]:
        validation = self.validator.validate(diagram, requirements)
        defects = list(validation.defects)

        plantuml_text, puml_defect = self._compile_plantuml(
            diagram,
            iteration=0,
        )
        if puml_defect is not None:
            defects.append(puml_defect)

        self.log.info("Running semantic reviewer...")
        count_llm_call()
        review = self.reviewer.run(
            requirement_text,
            requirements,
            matrix,
            diagram,
            existing_defects=defects,
        )
        defects.extend(review.defects)
        defects = self._deduplicate_defects(defects)
        return validation, review, defects, plantuml_text

    @staticmethod
    def _fallback_plantuml(sample_id: str, state: PipelineState) -> str:
        title = (state.diagram.title if state.diagram and state.diagram.title else sample_id).replace("\n", " ")
        return (
            "@startuml\n"
            f"title {title}\n"
            "start\n"
            "note right: Diagram generation failed; structural defects remain.\n"
            "stop\n"
            "@enduml\n"
        )

    def _compile_plantuml(
        self,
        diagram: ActivityDiagram | None,
        *,
        iteration: int,
    ) -> tuple[str, Defect | None]:
        if diagram is None:
            return "", Defect(
                id=f"PLANTUML-NODIAGRAM-{iteration}",
                category="SYNTAX",
                severity=Severity.CRITICAL,
                description="No activity diagram is available for PlantUML compilation.",
                node_ids=[], edge_ids=[], requirement_ids=[],
                evidence="diagram=None",
                suggested_action="Provide a valid ActivityDiagram before compilation.",
            )

        try:
            text = self.plantuml.render(diagram)
        except Exception as exc:
            return "", Defect(
                id=f"PLANTUML-COMPILE-{iteration}",
                category="SYNTAX",
                severity=Severity.HIGH,
                description="Activity Diagram could not be compiled to PlantUML.",
                node_ids=[], edge_ids=[], requirement_ids=[],
                evidence=str(exc),
                suggested_action="Repair the ActivityDiagram structure so PlantUML can be generated.",
            )

        if not text or not text.strip():
            return "", Defect(
                id=f"PLANTUML-EMPTY-{iteration}",
                category="SYNTAX",
                severity=Severity.CRITICAL,
                description="PlantUML generator returned an empty result.",
                node_ids=[], edge_ids=[], requirement_ids=[],
                evidence="PlantUML text was empty.",
                suggested_action="Ensure the final ActivityDiagram is valid and the compiler returns @startuml/@enduml.",
            )

        valid, message = self.plantuml_validator.validate(text)
        if not valid:
            return text, Defect(
                id=f"PLANTUML-INVALID-{iteration}",
                category="SYNTAX",
                severity=Severity.HIGH,
                description="Generated PlantUML failed syntax validation.",
                node_ids=[], edge_ids=[], requirement_ids=[],
                evidence=message,
                suggested_action="Repair the ActivityDiagram so the generated PlantUML passes syntax validation.",
            )
        return text, None

    # ============================================================
    # REPAIR ACCEPTANCE
    # ============================================================

    @staticmethod
    def _should_accept_repair(
        *,
        original_quality: float,
        repaired_quality: float,
        original_defect_count: int,
        repaired_defect_count: int,
        defects_fixed: int,
        defects_introduced: int,
        tolerance: float = 0.01,
    ) -> tuple[bool, str]:
        if repaired_defect_count > original_defect_count:
            return False, "DEFECT_COUNT_INCREASED"

        if defects_introduced > 0 and repaired_defect_count >= original_defect_count:
            return False, "NEW_DEFECT_WITHOUT_NET_REDUCTION"

        if repaired_defect_count < original_defect_count:
            return True, "FEWER_DEFECTS"

        if (
            repaired_defect_count == original_defect_count
            and defects_fixed > 0
            and defects_introduced == 0
            and repaired_quality > original_quality + tolerance
        ):
            return True, "SAME_DEFECT_COUNT_BUT_QUALITY_IMPROVED"

        return False, "NO_MEANINGFUL_IMPROVEMENT"

    @staticmethod
    def _is_deterministic_repair_batch(defects: list[Defect]) -> bool:
        return bool(defects) and all(
            defect.category in {"STRUCTURAL", "DECISION"}
            for defect in defects
        )

    # ============================================================
    # BEST CANDIDATE
    # ============================================================

    @staticmethod
    def _is_better_candidate(
        candidate: CandidateRecord,
        best: CandidateRecord | None,
    ) -> bool:
        if best is None:
            return True
        a = candidate.metrics
        b = best.metrics
        return (
            a.quality_score,
            -a.defect_count,
            a.requirement_coverage,
            a.correctness,
            a.completeness,
        ) > (
            b.quality_score,
            -b.defect_count,
            b.requirement_coverage,
            b.correctness,
            b.completeness,
        )

    # ============================================================
    # METRICS
    # ============================================================

    @staticmethod
    def _calculate_requirement_coverage(
        requirements: list[Requirement],
        diagram: ActivityDiagram,
    ) -> list[RequirementCoverage]:
        result = []
        for requirement in requirements:
            node_matches = [
                n for n in diagram.nodes if requirement.id in n.requirement_ids
            ]
            edge_matches = [
                e for e in diagram.edges if requirement.id in e.requirement_ids
            ]
            covered = bool(node_matches or edge_matches)
            if node_matches and edge_matches:
                coverage_type = "FULL"
            elif node_matches:
                coverage_type = "NODE"
            elif edge_matches:
                coverage_type = "EDGE"
            else:
                coverage_type = "NONE"

            evidence = []
            if node_matches:
                evidence.append("Nodes: " + ", ".join(n.id for n in node_matches))
            if edge_matches:
                evidence.append("Edges: " + ", ".join(e.id for e in edge_matches))

            result.append(
                RequirementCoverage(
                    requirement_id=requirement.id,
                    covered=covered,
                    coverage_type=coverage_type,
                    evidence="; ".join(evidence),
                )
            )
        return result

    @staticmethod
    def _coverage_score(coverage: list[RequirementCoverage]) -> float:
        if not coverage:
            return 1.0
        return sum(item.covered for item in coverage) / len(coverage)

    @staticmethod
    def _calculate_correctness(
        validation: ValidationResult,
        review: ReviewResult,
        defects: list[Defect],
    ) -> float:
        structural = validation.score
        semantic = review.semantic_score
        major = sum(
            1 for defect in defects
            if defect.severity in {Severity.CRITICAL, Severity.HIGH}
        )
        return max(0.0, min(1.0, 0.5 * structural + 0.5 * semantic - min(1.0, 0.15 * major)))

    @staticmethod
    def _calculate_completeness(
        requirements: list[Requirement],
        coverage: list[RequirementCoverage],
    ) -> float:
        if not requirements:
            return 1.0
        return sum(item.covered for item in coverage) / len(requirements)

    @staticmethod
    def _calculate_structural_validity(validation: ValidationResult) -> float:
        return max(0.0, min(1.0, validation.score))

    @staticmethod
    def _calculate_unsupported_behaviour_rate(
        review: ReviewResult,
        diagram: ActivityDiagram,
    ) -> float:
        relevant_nodes = [
            n for n in diagram.nodes
            if n.type.value in {"action", "decision", "object", "note"}
        ]
        if not relevant_nodes:
            return 0.0
        return max(
            0.0,
            min(1.0, len(review.unsupported_behaviors) / len(relevant_nodes)),
        )

    @staticmethod
    def _reduction_rate(previous_count: int, current_count: int) -> float:
        if previous_count <= 0:
            return 0.0
        return max(0.0, min(1.0, (previous_count - current_count) / previous_count))

    @staticmethod
    def _deduplicate_defects(defects: list[Defect]) -> list[Defect]:
        result = []
        seen = set()
        for defect in defects:
            key = defect_key(defect)
            if key in seen:
                continue
            seen.add(key)
            result.append(defect)
        return result

    @staticmethod
    def _filter_feedback_defects(
        proposed: list[Defect],
        actual: list[Defect],
    ) -> list[Defect]:
        """Only allow the feedback agent to reorder existing defects."""
        actual_by_id = {d.id: d for d in actual}
        actual_by_key = {defect_key(d): d for d in actual}
        selected = []
        seen = set()
        for defect in proposed:
            match = actual_by_id.get(defect.id) or actual_by_key.get(defect_key(defect))
            if match is None:
                continue
            key = defect_key(match)
            if key not in seen:
                seen.add(key)
                selected.append(match)
        return selected

    # ============================================================
    # REPAIR HISTORY / SUMMARY
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
        before_defect_count: int,
        after_defect_count: int,
        defects_fixed: int,
        defects_introduced: int,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "repair_type": repair.repair_type,
            "changed": repair.changed,
            "accepted": accepted,
            "reason": reason,
            "before_score": before_score,
            "after_score": after_score,
            "score_delta": after_score - before_score,
            "before_defect_count": before_defect_count,
            "after_defect_count": after_defect_count,
            "defects_fixed": defects_fixed,
            "defects_introduced": defects_introduced,
            "changes": repair.changes,
            "rationale": repair.rationale,
            "defect_ids": [d.id for d in selected_defects],
            "defect_categories": [d.category for d in selected_defects],
        }

    @staticmethod
    def _build_research_summary(state: PipelineState) -> dict[str, Any]:
        if not state.candidates:
            return {
                "candidate_count": 0,
                "best_iteration": None,
                "best_quality": 0.0,
                "initial_defects": 0,
                "final_defects": len(state.defects),
            }

        first = state.candidates[0]
        best = max(state.candidates, key=lambda c: c.metrics.quality_score)
        final = state.candidates[-1]
        initial_defects = first.metrics.defect_count
        final_defects = final.metrics.defect_count

        return {
            "candidate_count": len(state.candidates),
            "initial_iteration": first.iteration,
            "best_iteration": best.iteration,
            "best_quality_score": best.metrics.quality_score,
            "final_quality_score": final.metrics.quality_score,
            "initial_defects": initial_defects,
            "final_defects": final_defects,
            "overall_defect_reduction_rate": (
                max(0.0, min(1.0, (initial_defects - final_defects) / initial_defects))
                if initial_defects > 0
                else 0.0
            ),
            "best_requirement_coverage": best.metrics.requirement_coverage,
            "best_correctness": best.metrics.correctness,
            "best_completeness": best.metrics.completeness,
            "best_structural_validity": best.metrics.structural_validity,
            "best_unsupported_behaviour_rate": best.metrics.unsupported_behaviour_rate,
            "accepted_repairs": sum(r.get("accepted", False) for r in state.repair_history),
            "rejected_repairs": sum(not r.get("accepted", False) for r in state.repair_history),
            "total_defects_fixed": sum(r.get("defects_fixed", 0) for r in state.repair_history),
            "total_defects_introduced": sum(r.get("defects_introduced", 0) for r in state.repair_history),
        }
