from __future__ import annotations

import logging
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.feedback_agent import FeedbackAgent
from src.agents.generator_agent import GeneratorAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.consolidation_agent import ConsolidationAgent, apply_consolidation
from src.agents.outline_agent import OutlineAgent, OutlineRepairAgent
from src.generation.outline_renderer import OutlineRenderer
from src.models.outline import (
    ActivityOutline,
    copy_lanes,
    normalize_outline,
    outline_coverage,
    outline_defects,
    outline_listing,
    outline_to_diagram,
    requirements_for_outline,
    traced_requirement_ids,
)
from src.agents.requirement_agent import RequirementAgent
from src.document.chunker import DocumentChunker, estimate_tokens
from src.pipeline.extraction import extract_chunked
from src.pipeline.requirement_consolidation import consolidate_requirements
from src.agents.reviewer_agent import ReviewerAgent
from src.generation.ir_sanitizer import IRSanitizer
from src.generation.plantuml_generator import PlantUMLGenerator
from src.agents.lane_agent import ActorAgent, LaneMappingAgent
from src.agents.refinement_agent import RefinementAgent
from src.evaluation.semantic_metrics import label_similarity
from src.validation.structural.structural_validator import StructuralValidator
from src.generation.renderer import PlantUMLRenderer
from src.pipeline.lanes import (
    apply_lanes,
    canonical_actors,
    match_actor,
    spec_abbreviations,
    spec_qualified_actors,
    swimlane_instruction,
)
from src.llm import usage as llm_usage
from src.validation.reachability import reachable_node_ids
from src.llm.openai_client import OpenAIClient
from src.models.domain import (
    modelled_requirements,
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
    RepairResult,
)
from src.pipeline.defect_utils import defect_key, defect_signatures
from src.pipeline.repair_router import RepairRouter
from src.pipeline.scoring import candidate_score, quality_score
from src.validation.syntax.plantuml_validator import PlantUMLSyntaxValidator
from src.validation.validator import HybridValidator


# Deterministic defects that make a diagram wrong rather than improvable:
# broken structure or flow, PlantUML errors and requirements not shown.
BLOCKING_DEFECT_PREFIXES = (
    "D-STRUCT", "D-FLOW", "D-TERM", "D-CON", "D-DECISION", "D-SYN", "D-COV", "PLANTUML-",
)


def serious_count(defects: list[Defect]) -> int:
    return sum(d.severity in (Severity.HIGH, Severity.CRITICAL) for d in defects)


def blocking_defects(defects: list[Defect]) -> list[Defect]:
    return [d for d in defects if d.id.startswith(BLOCKING_DEFECT_PREFIXES)]


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

    # Repair rounds allowed after max_iterations while blocking defects remain.
    EXTRA_BLOCKING_ROUNDS = 3
    # Coverage defects repaired together in one outline repair.
    COVERAGE_BATCH = 15
    # An outline tracing fewer requirements than this is drawn again.
    MIN_OUTLINE_COVERAGE = 0.7
    # Above this size requirements are extracted chunk by chunk.
    CHUNKED_EXTRACTION_TOKENS = 2000

    def __init__(
        self,
        model: str | None = None,
        max_defects_per_repair: int = 3,
        llm: OpenAIClient | None = None,
        planner: str = "outline",
    ) -> None:
        # "outline": the diagram is planned, repaired and drawn as nested
        # blocks (if/while/fork), like an expert's diagram. "graph": the
        # Planning -> Generator -> Refinement agents build a free graph.
        if planner not in ("outline", "graph"):
            raise ValueError(f"Unknown planner: {planner}")
        self.planner = planner
        # An injected client lets the hierarchical pipeline share one client
        # (and its connection pool) across many concurrently running modules.
        self.llm = llm or OpenAIClient(model=model)

        self.requirement_agent = RequirementAgent(self.llm)
        self.consolidation_agent = ConsolidationAgent(self.llm)
        self.outline_agent = OutlineAgent(self.llm)
        self.outline_repair_agent = OutlineRepairAgent(self.llm)
        self.outline_renderer = OutlineRenderer()
        self.actor_agent = ActorAgent(self.llm)
        self.refinement_agent = RefinementAgent(self.llm)
        self.lane_agent = LaneMappingAgent(self.llm)
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
        requirements: list[Requirement] | None = None,
        render: bool = True,
    ) -> PipelineState:
        """Run the full agent workflow for one requirement text.

        ``requirements`` may carry an already-extracted requirement set (used
        by the hierarchical long-document pipeline, which extracts per chunk
        and then runs this workflow per module); the Requirement Agent is then
        skipped. ``render=False`` skips writing/rendering the image next to
        ``output_dir`` (the caller renders instead).
        """
        pipeline_start = time.perf_counter()
        max_iterations = max(0, max_iterations)
        llm_calls = 0
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        run_dir = output_path / (
            f"{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        # Abbreviations the specification defines ("Peering Agent (PA)").
        self._abbreviations = spec_abbreviations(requirement_text)
        self._evaluation_cache = {}
        # Token accounting scope for this run (observation only).
        usage_run_id = uuid.uuid4().hex
        llm_usage.set_run(usage_run_id)
        llm_usage.set_phase("setup")

        def save_stage(name: str, value: Any) -> None:
            stage_path = run_dir / f"{name}.json"
            if hasattr(value, "model_dump"):
                payload = value.model_dump(mode="json")
            else:
                payload = value
            stage_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

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
        if requirements is not None:
            state.requirements = [r.model_copy(deep=True) for r in requirements]
            state.requirement_matrix = RequirementMatrix(items=[])
            save_stage(
                "01_requirement_agent",
                {
                    "source": "provided",
                    "requirements": [r.model_dump(mode="json") for r in state.requirements],
                },
            )
        else:
            self.log.info("Running requirement agent...")
            if estimate_tokens(requirement_text) > self.CHUNKED_EXTRACTION_TOKENS:
                # One call over a long text stops partway through it: extract
                # per section-aware chunk, then model one diagram from all of it.
                chunker = DocumentChunker(max_tokens=self.CHUNKED_EXTRACTION_TOKENS)
                chunks = chunker.chunk(requirement_text)
                per_chunk, errors = extract_chunked(
                    self.requirement_agent, chunker, chunks, sample_id, log=self.log
                )
                for _ in per_chunk:
                    call_llm()
                state.requirements = consolidate_requirements(per_chunk)
                state.requirement_matrix = RequirementMatrix(items=[])
                # Sections restate the same step and explain how it works:
                # keep one requirement per step so the diagram is drawn at the
                # level of the specification's main steps.
                try:
                    call_llm()
                    verdicts = self.consolidation_agent.run(state.requirements)
                    merged = apply_consolidation(state.requirements, verdicts)
                    save_stage("01a_consolidation_agent", verdicts)
                    self.log.info("Consolidation set aside %d restated or explanatory requirements.", merged)
                except Exception as exc:
                    self.log.warning("Consolidation agent failed (%s); all requirements are kept.", exc)
                save_stage(
                    "01_requirement_agent",
                    {
                        "source": f"{len(chunks)} chunks",
                        "errors": errors,
                        "requirements": [r.model_dump(mode="json") for r in state.requirements],
                    },
                )
            else:
                call_llm()
                extracted = self.requirement_agent.run(requirement_text)
                state.requirements = extracted.requirements
                state.requirement_matrix = extracted.matrix
                save_stage(
                    "01_requirement_agent",
                    extracted,
                )
        self.log.info("Extracted %d requirements.", len(state.requirements))

        # --------------------------------------------------------
        # 1b. Swimlanes: the actors are chosen once, before planning, so
        #     every agent uses the same lanes. (Hierarchical modules get
        #     them from the decomposition instead.)
        # --------------------------------------------------------
        actors: list[str] = []
        spec_text = requirement_text
        if requirements is None:
            self.log.info("Running actor agent...")
            call_llm()
            proposed: list[str] = []
            try:
                proposed = self.actor_agent.run(requirement_text)
                actors = spec_qualified_actors(
                    canonical_actors(proposed),
                    requirement_text,
                    self._abbreviations,
                )
                if len(actors) < 2 <= len(proposed):
                    # Filtering must not leave one lane for an interaction
                    # between several participants.
                    self.log.warning("Lane filter kept %s of %s; using the proposed lanes.", actors, proposed)
                    actors = list(dict.fromkeys(" ".join(a.split()) for a in proposed))
            except Exception as exc:
                self.log.warning("Actor agent failed (%s); lanes are left to the planner.", exc)
            if actors:
                save_stage("01b_actor_agent", {"actors": actors, "proposed": proposed})
                self.log.info("Swimlanes: %s", ", ".join(actors))
                requirement_text = "\n\n".join([requirement_text, "\n".join(swimlane_instruction(actors))])

        # --------------------------------------------------------
        # 2-3. Planning and generation
        # --------------------------------------------------------
        outline: ActivityOutline | None = None
        if self.planner == "outline":
            self.log.info("Running outline agent...")
            call_llm()
            outline = normalize_outline(
                self.outline_agent.run(spec_text, state.requirements, actors)
            )
            # An outline that leaves most requirements out is drawn again,
            # told which ones it missed; the better of the two is kept.
            coverage = outline_coverage(outline, state.requirements)
            if coverage < self.MIN_OUTLINE_COVERAGE:
                missed = self._untraced(outline, state.requirements)
                self.log.warning(
                    "Outline covers %.0f%% of the requirements; drawing it again.", 100 * coverage,
                )
                call_llm()
                try:
                    retry = normalize_outline(self.outline_agent.run(
                        spec_text, state.requirements, actors,
                        feedback=f"A previous outline covered only {100 * coverage:.0f}% of the "
                                 f"behavioural requirements. It missed: {', '.join(missed)}. Draw the "
                                 "whole behaviour of the specification.",
                    ))
                    if outline_coverage(retry, state.requirements) > coverage:
                        outline = retry
                except Exception as exc:
                    self.log.warning("Second outline failed (%s); keeping the first.", exc)
            state.requirements, set_aside = requirements_for_outline(state.requirements, outline)
            state.diagram = outline_to_diagram(outline)
            if actors:
                # Lanes the outline invented ("Agents") become the chosen actors.
                self._harmonize_lanes(state.diagram, actors, call_llm)
                copy_lanes(outline, state.diagram)
            save_stage("02_outline_agent", {"outline": outline.model_dump(mode="json"), "not_drawn_accepted": set_aside})
            self.log.info(
                "Outline: %d nodes, %d edges; %d requirements left out as restatements or context.",
                len(state.diagram.nodes), len(state.diagram.edges), len(set_aside),
            )
        else:
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
            save_stage("02_planning_agent", state.plan)
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
            save_stage("03_generator_agent", state.diagram)
            self.log.info(
                "Generated diagram: %d nodes, %d edges.",
                len(state.diagram.nodes),
                len(state.diagram.edges),
            )

            # --------------------------------------------------------
            # 3b. Modelling refinement: one review of the whole draft (merge
            #     duplicated alternatives, draw shared and actor steps once).
            #     Kept only if it loses no coverage and adds no structural defect.
            # --------------------------------------------------------
            self.log.info("Running refinement agent...")
            call_llm()
            try:
                refinement = self.refinement_agent.run(requirement_text, state.requirements, state.diagram)
                refined = IRSanitizer.sanitize(refinement.diagram)
                carried = self._carry_over_traces(state.diagram, refined)
                accepted, reason = self._accept_refinement(state.diagram, refined, state.requirements)
                if carried:
                    reason += f"; {carried} requirement traces carried over to merged steps"
                save_stage(
                    "03b_refinement_agent",
                    {"changes": refinement.changes, "accepted": accepted, "reason": reason,
                     "diagram": refined.model_dump(mode="json")},
                )
                self.log.info("Refinement %s (%s): %d changes.", "accepted" if accepted else "rejected",
                              reason, len(refinement.changes))
                if accepted:
                    state.diagram = refined
            except Exception as exc:
                self.log.warning("Refinement agent failed (%s); keeping the first draft.", exc)

        # --------------------------------------------------------
        # 3c. The diagram as it enters validation and repair, kept to
        #     compare with the final diagram.
        # --------------------------------------------------------
        initial_plantuml, _ = self._compile_candidate(state.diagram, outline, iteration=0)
        (run_dir / "initial.puml").write_text(initial_plantuml, encoding="utf-8")
        outlines: dict[int, ActivityOutline] = {}
        unchanged_once = False

        best_quality = -1.0
        best_iteration = 0
        best_candidate: CandidateRecord | None = None
        previous_signatures: set[str] | None = None
        previous_defect_count: int | None = None

        # --------------------------------------------------------
        # 4. Iterative evaluation / repair loop
        # --------------------------------------------------------
        # Defects that make the diagram wrong (not just improvable) get extra
        # repair rounds after max_iterations; see blocking_defects().
        extra_rounds = self.EXTRA_BLOCKING_ROUNDS if max_iterations > 0 else 0
        last_round = max_iterations + extra_rounds

        for iteration in range(last_round + 1):
            state.iteration = iteration
            llm_usage.set_phase(f"iteration_{iteration:02d}")
            iteration_start = time.perf_counter()
            self.log.info("Validation iteration %d", iteration)

            validation, review, defects, plantuml_text = self._evaluate_candidate(
                requirement_text=requirement_text,
                requirements=state.requirements,
                matrix=state.requirement_matrix,
                diagram=state.diagram,
                count_llm_call=call_llm,
                outline=outline,
            )
            if outline is not None:
                outlines[iteration] = outline.model_copy(deep=True)

            state.validation = validation
            state.review = review
            state.defects = defects
            save_stage(
                f"04_iteration_{iteration:02d}_validation",
                {
                    "validation": validation.model_dump(mode="json"),
                    "review": review.model_dump(mode="json"),
                    "defects": [d.model_dump(mode="json") for d in defects],
                    "plantuml": plantuml_text,
                },
            )
            (run_dir / f"04_iteration_{iteration:02d}.puml").write_text(plantuml_text, encoding="utf-8")

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
            if all(d.severity == Severity.LOW for d in defects):
                # Rewording for minor findings drifts from the plan without
                # making the diagram more correct.
                self.log.info("Only minor (LOW) findings remain; stopping repairs.")
                break

            blocking = blocking_defects(defects)
            blocking_round = iteration >= max_iterations
            if blocking_round:
                if not blocking or iteration >= last_round:
                    if blocking:
                        self.log.warning(
                            "Blocking defects remain after %d extra rounds: %s",
                            extra_rounds, [d.id for d in blocking],
                        )
                    self.log.warning("Maximum repair iterations reached.")
                    break
                self.log.info(
                    "Extra repair round %d for blocking defects: %s",
                    iteration - max_iterations + 1, [d.id for d in blocking],
                )

            # ----------------------------------------------------
            # 5. Feedback / defect prioritization
            # ----------------------------------------------------
            if blocking_round:
                prioritized = self.repair_router.select_defects(blocking)
            else:
                self.log.info("Running feedback agent...")
                call_llm()
                feedback_result = self.feedback.run(defects)
                save_stage(
                    f"05_iteration_{iteration:02d}_feedback_agent",
                    feedback_result,
                )
                prioritized = self._filter_feedback_defects(
                    feedback_result.prioritized_defects,
                    defects,
                )
                prioritized = self.repair_router.select_defects(
                    prioritized or defects
                )

            if outline is not None:
                # Missing traces are one kind of fix (trace, add or set aside
                # a requirement): repair them together, not three per round.
                missing = [d for d in defects if d.id.startswith("D-COV")]
                extra = [d for d in missing if d not in prioritized][: self.COVERAGE_BATCH]
                prioritized = prioritized + extra

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

            repaired_outline: ActivityOutline | None = None
            repaired_requirements = state.requirements
            if outline is not None:
                repair, repaired_outline, repaired_requirements = self._repair_outline(
                    spec_text, state.requirements, outline, prioritized, actors,
                )
            else:
                repair = self.repair_router.repair(
                    requirement_text,
                    state.requirements,
                    state.diagram,
                    prioritized,
                )
            save_stage(
                f"06_iteration_{iteration:02d}_repair_agent",
                repair,
            )
            if outline is not None or getattr(self.repair_router, "last_used_llm", False):
                call_llm()

            if not repair.changed and outline is not None and not unchanged_once:
                # The LLM sometimes returns the outline as it was; try once more.
                unchanged_once = True
                self.log.warning("Repair made no changes; trying again.")
                continue
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

            repaired_diagram = (
                repair.diagram if repaired_outline is not None else IRSanitizer.sanitize(repair.diagram)
            )
            save_stage(
                f"07_iteration_{iteration:02d}_repaired_diagram",
                repaired_diagram,
            )
            if repaired_outline is not None:
                save_stage(f"07_iteration_{iteration:02d}_repaired_outline", repaired_outline)

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
                requirements=repaired_requirements,
                matrix=state.requirement_matrix,
                diagram=repaired_diagram,
                count_llm_call=call_llm,
                outline=repaired_outline,
            )

            regression_coverage = self._calculate_requirement_coverage(
                repaired_requirements, repaired_diagram
            )
            regression_coverage_score = self._coverage_score(regression_coverage)
            regression_correctness = self._calculate_correctness(
                regression_validation,
                regression_review,
                regression_defects,
            )
            regression_completeness = self._calculate_completeness(
                repaired_requirements,
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
                original_blocking=defect_signatures(blocking_defects(defects)),
                repaired_blocking=defect_signatures(blocking_defects(regression_defects)),
                original_serious=serious_count(defects),
                repaired_serious=serious_count(regression_defects),
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
                if repaired_outline is not None:
                    outline = repaired_outline
                    state.requirements = repaired_requirements
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
                if iteration >= last_round:
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

        final_outline = outlines.get(best_iteration) if outline is not None else None
        llm_usage.set_phase("finalise")
        if actors and state.diagram is not None:
            self._harmonize_lanes(state.diagram, actors, call_llm)
            if final_outline is not None:
                copy_lanes(final_outline, state.diagram)
        if final_outline is not None:
            state.outline = final_outline.model_dump(mode="json")

        # --------------------------------------------------------
        # 10. Final PlantUML with explicit non-empty check
        # --------------------------------------------------------
        state.final_plantuml, final_puml_defect = self._compile_candidate(
            state.diagram,
            final_outline,
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
        render_info: dict[str, Any] = {}

        if render and initial_plantuml:
            # Before/after comparison: the diagram as first drawn.
            try:
                (output_path / f"{sample_id}_before_repair.puml").write_text(initial_plantuml, encoding="utf-8")
                self.renderer.render(initial_plantuml, output_path, name=f"{sample_id}_before_repair")
            except Exception as exc:
                self.log.warning("Rendering the initial diagram failed: %s", exc)

        if not render:
            render_info = {"rendered": False, "skipped": True}
        elif state.final_plantuml:
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
        state.metrics["planner"] = self.planner
        if state.candidates:
            first = state.candidates[0]
            state.metrics["before_repair"] = {
                "plantuml_file": str(run_dir / "initial.puml"),
                "defect_count": first.metrics.defect_count,
                "blocking_defects": len(blocking_defects(first.defects)),
                "requirement_coverage": first.metrics.requirement_coverage,
                "quality_score": first.metrics.quality_score,
            }
            state.metrics["after_repair"] = {
                "plantuml_file": str(run_dir / "final.puml"),
                "iteration": best_iteration,
                "defect_count": len(state.defects),
                "blocking_defects": len(blocking_defects(state.defects)),
                "requirement_coverage": best_candidate.metrics.requirement_coverage if best_candidate else None,
                "quality_score": best_quality,
            }
        state.metrics["token_usage"] = self._token_usage(usage_run_id, state.metrics.get("candidate_scores", []))
        state.metrics["run_output_dir"] = str(run_dir)
        save_stage("08_final_state", state)
        (run_dir / "final.puml").write_text(
            state.final_plantuml,
            encoding="utf-8",
        )

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
        outline: ActivityOutline | None = None,
    ) -> tuple[ValidationResult, ReviewResult, list[Defect], str]:
        # The same candidate is evaluated once: re-reviewing an unchanged
        # diagram gives different findings and makes iterations incomparable.
        cache_key = json.dumps([
            diagram.model_dump(mode="json"),
            outline.model_dump(mode="json") if outline is not None else None,
            [(r.id, r.type.value) for r in requirements],
        ], sort_keys=True)
        cache = self.__dict__.setdefault("_evaluation_cache", {})
        if cache_key in cache:
            validation, review, defects, text = cache[cache_key]
            return (validation.model_copy(deep=True), review.model_copy(deep=True),
                    [d.model_copy(deep=True) for d in defects], text)
        result = self._evaluate_uncached(
            requirement_text=requirement_text, requirements=requirements, matrix=matrix,
            diagram=diagram, count_llm_call=count_llm_call, outline=outline,
        )
        cache[cache_key] = result
        validation, review, defects, text = result
        return (validation.model_copy(deep=True), review.model_copy(deep=True),
                [d.model_copy(deep=True) for d in defects], text)

    def _evaluate_uncached(
        self,
        *,
        requirement_text: str,
        requirements: list[Requirement],
        matrix: RequirementMatrix,
        diagram: ActivityDiagram,
        count_llm_call,
        outline: ActivityOutline | None = None,
    ) -> tuple[ValidationResult, ReviewResult, list[Defect], str]:
        validation = self.validator.validate(diagram, requirements)
        defects = list(validation.defects)

        if outline is not None:
            defects.extend(outline_defects(outline))
        plantuml_text, puml_defect = self._compile_candidate(diagram, outline, iteration=0)
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
            diagram_text=outline_listing(outline) if outline is not None else None,
        )
        defects.extend(review.defects)
        defects = self._deduplicate_defects(defects)
        return validation, review, defects, plantuml_text

    def _token_usage(self, run_id: str, candidate_scores: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokens and cost of this run, per phase and agent; adds each
        iteration's own tokens and cost to its candidate score entry."""
        tracker = getattr(getattr(self, "llm", None), "usage", None)
        if tracker is None:
            return {}
        result = llm_usage.report(tracker.records(run=run_id))
        for entry in candidate_scores:
            phase = result["by_phase"].get(f"iteration_{entry['iteration']:02d}", {})
            entry["iteration_tokens"] = phase.get("total_tokens", 0)
            entry["iteration_cost_usd"] = phase.get("cost_usd")
        return result

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

    @staticmethod
    def _carry_over_traces(draft: ActivityDiagram, refined: ActivityDiagram, threshold: float = 0.5) -> int:
        """Re-attach requirement IDs that merging steps dropped.

        A requirement traced in the draft but not in the refinement moves to
        the refined node of the same type whose label is most like the draft
        node that carried it. Below ``threshold`` it is left missing, so a
        step the refinement really deleted still fails the coverage check.
        """
        refined_ids = {rid for e in [*refined.nodes, *refined.edges] for rid in e.requirement_ids}
        moved = 0
        for node in draft.nodes:
            for rid in node.requirement_ids:
                if rid in refined_ids:
                    continue
                candidates = [n for n in refined.nodes if n.type == node.type]
                if not candidates:
                    continue
                best = max(candidates, key=lambda n: label_similarity(n.label, node.label))
                if label_similarity(best.label, node.label) >= threshold:
                    best.requirement_ids.append(rid)
                    refined_ids.add(rid)
                    moved += 1
        return moved

    def _accept_refinement(
        self,
        original: ActivityDiagram,
        refined: ActivityDiagram,
        requirements: list[Requirement],
    ) -> tuple[bool, str]:
        try:
            self.plantuml.render(refined)
        except Exception as exc:
            return False, f"does not compile: {exc}"

        def serious(diagram: ActivityDiagram) -> int:
            return sum(
                1 for d in StructuralValidator().validate(diagram).defects
                if d.severity in (Severity.HIGH, Severity.CRITICAL)
            )

        if serious(refined) > serious(original):
            return False, "more structural defects"

        def covered(diagram: ActivityDiagram) -> int:
            return sum(c.covered for c in self._calculate_requirement_coverage(requirements, diagram))

        if covered(refined) < covered(original):
            return False, f"coverage {covered(original)} -> {covered(refined)}"
        return True, "no coverage or structural loss"

    def _harmonize_lanes(self, diagram: ActivityDiagram, actors: list[str], call_llm) -> None:
        """Map lane names the agents invented onto the chosen actors."""
        steps: dict[str, list[str]] = {}
        for node in diagram.nodes:
            if node.lane and node.lane.strip():
                steps.setdefault(" ".join(node.lane.split()), []).append(node.label)
        abbreviations = getattr(self, "_abbreviations", {})
        mapping: dict[str, str | None] = {lane: match_actor(lane, actors, abbreviations) for lane in steps}
        unresolved = {lane: labels for lane, labels in steps.items() if mapping[lane] is None}
        if unresolved:
            call_llm()
            try:
                mapping.update(self.lane_agent.run(unresolved, actors, abbreviations))
            except Exception as exc:
                self.log.warning("Lane mapping failed (%s); unmatched lanes follow the preceding step.", exc)
        apply_lanes(diagram, mapping)

    def _compile_candidate(
        self,
        diagram: ActivityDiagram | None,
        outline: ActivityOutline | None,
        *,
        iteration: int,
    ) -> tuple[str, Defect | None]:
        """PlantUML of a candidate: drawn from its outline when it has one
        (blocks exactly as planned), otherwise compiled from the IR."""
        if outline is not None:
            try:
                text = self.outline_renderer.render(outline)
            except Exception as exc:
                self.log.warning("Outline could not be drawn (%s); compiling the IR.", exc)
            else:
                valid, message = self.plantuml_validator.validate(text)
                if valid:
                    return text, None
                self.log.info("Outline drawing failed PlantUML (%s); compiling the IR.", message)
        return self._compile_plantuml(diagram, iteration=iteration)

    @staticmethod
    def _untraced(outline: ActivityOutline, requirements: list[Requirement]) -> list[str]:
        adjusted, _ = requirements_for_outline(requirements, outline)
        traced = traced_requirement_ids(outline)
        return [r.id for r in modelled_requirements(adjusted) if r.id not in traced]

    def _repair_outline(
        self,
        spec_text: str,
        requirements: list[Requirement],
        outline: ActivityOutline,
        defects: list[Defect],
        actors: list[str],
    ) -> tuple[RepairResult, ActivityOutline | None, list[Requirement]]:
        """Repairs by editing the outline; returns the repair, the new outline
        and the requirements with its not-drawn entries applied."""
        self.log.info("Running outline repair agent...")
        try:
            result = self.outline_repair_agent.run(spec_text, requirements, outline, defects, actors)
        except Exception as exc:
            self.log.warning("Outline repair failed: %s", exc)
            return RepairResult(changed=False, repair_type="OUTLINE", diagram=outline_to_diagram(outline)), None, requirements
        repaired = normalize_outline(result.outline)
        if not repaired.not_drawn and outline.not_drawn:
            repaired.not_drawn = list(outline.not_drawn)
        if not repaired.title:
            repaired.title = outline.title
        changed = repaired.model_dump(exclude={"not_drawn"}) != outline.model_dump(exclude={"not_drawn"}) or (
            repaired.not_drawn != outline.not_drawn
        )
        repaired_requirements, _ = requirements_for_outline(requirements, repaired)
        diagram = outline_to_diagram(repaired)
        if actors:
            self._harmonize_lanes(diagram, actors, lambda: None)
            copy_lanes(repaired, diagram)
        return (
            RepairResult(
                changed=changed,
                repair_type="OUTLINE",
                changes=result.changes,
                rationale="Outline edited for the selected defects.",
                diagram=diagram,
            ),
            repaired,
            repaired_requirements,
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
            # PlantUML cannot draw some structured loops (e.g. a while body
            # that ends in another swimlane with a jump or stop). The same IR
            # drawn with plain branching usually renders; keep it if it does.
            try:
                fallback = self.plantuml.render(diagram, structured_loops=False)
            except Exception:
                fallback = ""
            if fallback and fallback != text and self.plantuml_validator.validate(fallback)[0]:
                self.log.info("PlantUML could not draw the structured loops (%s); using plain branching.", message)
                return fallback, None
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
        original_blocking: set[str] | None = None,
        repaired_blocking: set[str] | None = None,
        original_serious: int | None = None,
        repaired_serious: int | None = None,
    ) -> tuple[bool, str]:
        # Fixing a blocking defect (the diagram was wrong) outweighs new
        # reviewer findings, as long as it adds no blocking defect of its own
        # and either the overall quality or the number of serious (HIGH or
        # CRITICAL) defects does not get worse. (The reviewer's score varies
        # between calls; the serious-defect count is the steadier signal.)
        if (
            original_blocking is not None
            and repaired_blocking is not None
            and len(repaired_blocking) < len(original_blocking)
            and not (repaired_blocking - original_blocking)
            and (
                repaired_quality >= original_quality - tolerance
                or (
                    original_serious is not None
                    and repaired_serious is not None
                    and repaired_serious < original_serious
                )
            )
        ):
            return True, "FEWER_BLOCKING_DEFECTS"
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
        # A candidate without blocking defects (unreachable nodes, broken
        # flow, untraced requirements, ...) beats any candidate with them;
        # then fewer serious findings. The reviewer's scores vary between
        # calls, so they only break ties, and a full tie keeps the earlier
        # (less edited) candidate.
        def key(record: CandidateRecord) -> tuple:
            severities = [d.severity for d in record.defects]
            return (
                -len(blocking_defects(record.defects)),
                -sum(s in (Severity.HIGH, Severity.CRITICAL) for s in severities),
                -sum(s == Severity.MEDIUM for s in severities),
                record.metrics.requirement_coverage,
                record.metrics.quality_score,
            )

        return key(candidate) > key(best)

    # ============================================================
    # METRICS
    # ============================================================

    @staticmethod
    def _calculate_requirement_coverage(
        requirements: list[Requirement],
        diagram: ActivityDiagram,
    ) -> list[RequirementCoverage]:
        result = []
        # Unreachable elements are never drawn and do not count.
        reachable = reachable_node_ids(diagram)
        for requirement in modelled_requirements(requirements):
            node_matches = [
                n for n in diagram.nodes
                if requirement.id in n.requirement_ids and n.id in reachable
            ]
            edge_matches = [
                e for e in diagram.edges
                if requirement.id in e.requirement_ids and e.source in reachable
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
        requirements = modelled_requirements(requirements)
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
