from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.agents.decomposition_agent import DecompositionAgent
from src.agents.lane_agent import LaneMappingAgent
from src.agents.requirement_agent import RequirementAgent
from src.document.chunker import DocumentChunker, estimate_tokens
from src.generation.ir_sanitizer import IRSanitizer
from src.generation.plantuml_generator import PlantUMLGenerator
from src.generation.plantuml_tool import DEFAULT_LIMIT_SIZE
from src.generation.renderer import PlantUMLRenderer
from src.llm import usage as llm_usage
from src.llm.openai_client import OpenAIClient
from src.models.domain import (
    modelled_requirements,
    ActivityModule,
    DecompositionPlan,
    DocumentChunk,
    HierarchicalState,
    ModuleResult,
    Requirement,
)
from src.pipeline.composer import build_overview_diagram, compose_full_diagram, split_into_parts
from src.pipeline.lanes import apply_lanes, match_actor, swimlane_instruction
from src.reporting.viewer import build_viewer
from src.pipeline.decomposition import (
    normalize_decomposition,
    section_decomposition,
)
from src.pipeline.orchestrator import MultiAgentPipeline
from src.pipeline.requirement_consolidation import consolidate_requirements
from src.reporting.report import build_hierarchical_report
from src.validation.structural.structural_validator import StructuralValidator
from src.validation.syntax.plantuml_validator import PlantUMLCheck, PlantUMLSyntaxValidator

# Smallest "scale" applied to fit a diagram within PlantUML's default PNG
# size limit; below it the text is too small to read and the SVG is better.
MIN_FIT_SCALE = 0.6


class HierarchicalPipeline:
    """Long-document workflow (tens of pages of requirements).

    1. Chunk the document on section/paragraph boundaries.
    2. Requirement Agent per chunk (in parallel); consolidate into R1..Rn.
    3. Decomposition Agent: partition requirements into modules
       (sub-activities) plus the control flow between them.
    4. For every module, the standard agent workflow runs on that module's
       requirements only: Planning -> Generator -> Validator -> Semantic
       Reviewer -> Feedback -> Repair, with accept/rollback. Modules run
       in parallel.
    5. Overview diagram (one action per module) and a full diagram with
       every module inlined; both are validated deterministically.
    6. Global requirement coverage check and a Markdown report.

    Prompt size depends on the chunk/module size, not on document length.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        llm: OpenAIClient | None = None,
        chunk_tokens: int = 2500,
        overlap_tokens: int = 150,
        max_requirements_per_module: int = 25,
        max_workers: int = 4,
        max_defects_per_repair: int = 2,
        decomposition_token_budget: int = 60_000,
        pipeline_factory: Callable[[], MultiAgentPipeline] | None = None,
    ) -> None:
        self.llm = llm or OpenAIClient(model=model)
        self.chunker = DocumentChunker(max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        self.requirement_agent = RequirementAgent(self.llm)
        self.decomposition_agent = DecompositionAgent(self.llm)
        self.max_requirements_per_module = max(3, max_requirements_per_module)
        self.max_workers = max(1, max_workers)
        self.decomposition_token_budget = decomposition_token_budget
        self.pipeline_factory = pipeline_factory or (
            lambda: MultiAgentPipeline(llm=self.llm, max_defects_per_repair=max_defects_per_repair)
        )
        self.validator = StructuralValidator()
        self.syntax = PlantUMLSyntaxValidator()
        self.lane_agent = LaneMappingAgent(self.llm)
        self.renderer = PlantUMLRenderer()
        self.log = logging.getLogger(self.__class__.__name__)

    # ============================================================
    # MAIN
    # ============================================================

    def run(
        self,
        sample_id: str,
        requirement_text: str,
        max_iterations: int = 3,
        output_dir: str | Path = "outputs",
        document_title: str = "",
    ) -> HierarchicalState:
        started = time.perf_counter()
        output_path = Path(output_dir)
        run_dir = output_path / f"{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_dir.mkdir(parents=True, exist_ok=False)
        title = document_title or sample_id

        state = HierarchicalState(
            sample_id=sample_id,
            document_tokens=estimate_tokens(requirement_text),
        )
        llm_calls = 0
        # Token accounting (observation only): document-level calls carry no run.
        usage_tracker = getattr(getattr(self, "llm", None), "usage", None)
        usage_start = usage_tracker.count() if usage_tracker is not None else 0
        llm_usage.set_run(None)
        llm_usage.set_phase(None)

        # ---------------- 1. chunking ----------------
        state.chunks = self.chunker.chunk(requirement_text)
        self._save(run_dir / "00_chunks.json", state.chunks)
        self.log.info(
            "Document ~%d tokens split into %d chunks.",
            state.document_tokens,
            len(state.chunks),
        )

        # ---------------- 2. requirement extraction ----------------
        per_chunk, errors = self._extract(state.chunks, title)
        llm_calls += len(per_chunk)
        state.extraction_errors = errors
        state.requirements = consolidate_requirements(per_chunk)
        self._save(run_dir / "01_requirements.json", state.requirements)
        self.log.info("Consolidated %d requirements.", len(state.requirements))
        if not state.requirements:
            raise RuntimeError(
                "No requirements could be extracted from the document. "
                f"Extraction errors: {errors}"
            )

        # ---------------- 3. decomposition ----------------
        state.decomposition, used_llm = self._decompose(state.requirements, title)
        llm_calls += int(used_llm)
        self._save(run_dir / "02_decomposition.json", state.decomposition)
        self.log.info("Decomposed into %d modules.", len(state.decomposition.modules))

        # ---------------- 4. per-module agent workflow ----------------
        requirement_by_id = {r.id: r for r in state.requirements}
        modules_dir = run_dir / "modules"
        modules_dir.mkdir()

        def run_module(module: ActivityModule) -> ModuleResult:
            return self._run_module(
                module,
                state.decomposition,
                requirement_by_id,
                max_iterations,
                modules_dir,
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            state.modules = list(pool.map(run_module, state.decomposition.modules))
        llm_calls += sum(int(m.metrics.get("llm_calls", 0)) for m in state.modules)

        # ---------------- 4b. one set of swimlanes ----------------
        lane_mapping, used_llm = self._harmonize_lanes(state)
        llm_calls += int(used_llm)

        # ---------------- 5. overview + full diagram ----------------
        # Every diagram is checked with PlantUML before it is written: a
        # syntax error is reported and the text is saved as <name>.invalid.puml
        # instead of <name>.puml.
        compile_errors: dict[str, str] = {}
        checks: dict[str, dict[str, Any]] = {}
        overview = build_overview_diagram(state.decomposition)
        state.overview_diagram = overview
        state.overview_plantuml = self._compile_checked(
            overview, "overview", run_dir / "overview", compile_errors, checks,
        )

        full = compose_full_diagram(
            overview,
            {m.module.id: m.diagram for m in state.modules},
            title=state.decomposition.system_name or title,
        )
        state.full_diagram = IRSanitizer.sanitize(full)
        state.full_validation = self.validator.validate(state.full_diagram)
        state.full_plantuml = self._compile_checked(
            state.full_diagram, "full", run_dir / "full", compile_errors, checks,
        )

        # The full diagram cut into one linked part per module: each part is
        # small enough to read and keeps its swimlanes.
        parts = split_into_parts(state.full_diagram, state.decomposition)
        part_files: dict[int, tuple[str, str]] = {}
        for part in parts:
            key = f"part_{part.number:02d}"
            stem = f"{key}_{_slug(part.name)}"
            text = self._compile_checked(
                part.diagram, key, run_dir / "parts" / stem, compile_errors, checks, part.links,
            )
            if text:
                part_files[part.number] = (stem, text)

        def check_module(module: ModuleResult) -> tuple[ModuleResult, PlantUMLCheck | None]:
            return module, self.syntax.check(module.plantuml) if module.plantuml else None

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            module_checks = list(pool.map(check_module, state.modules))
        for module, result in module_checks:
            if result is not None:
                name = f"{module.module.id}_{_slug(module.module.name)}"
                module.plantuml, result, note = self._fit(module.plantuml, result)
                self._record_check(
                    module.module.id, module.plantuml, result, run_dir / "modules" / name,
                    compile_errors, checks, note,
                )

        # ---------------- 6. coverage + report ----------------
        covered = {
            rid
            for m in state.modules
            if m.diagram is not None
            for element in [*m.diagram.nodes, *m.diagram.edges]
            for rid in element.requirement_ids
        }
        modelled = modelled_requirements(state.requirements)
        state.uncovered_requirement_ids = [r.id for r in modelled if r.id not in covered]

        renders = {
            "overview": self._render(state.overview_plantuml, run_dir, f"{sample_id}_overview"),
            "full": self._render(state.full_plantuml, run_dir, f"{sample_id}_full"),
        }
        for module in state.modules:
            if module.plantuml and module.module.id not in compile_errors:
                renders[module.module.id] = self._render(
                    module.plantuml,
                    run_dir / "modules",
                    f"{module.module.id}_{_slug(module.module.name)}",
                )
        for number, (stem, text) in part_files.items():
            renders[f"part_{number:02d}"] = self._render(text, run_dir / "parts", stem)

        total = len(modelled)
        state.metrics = {
            "document_tokens": state.document_tokens,
            "chunk_count": len(state.chunks),
            "requirement_count": len(state.requirements),
            "non_behavioural_requirements": len(state.requirements) - total,
            "module_count": len(state.modules),
            "failed_modules": [m.module.id for m in state.modules if m.error],
            "requirement_coverage": (total - len(state.uncovered_requirement_ids)) / total if total else 1.0,
            "remaining_module_defects": sum(len(m.remaining_defects) for m in state.modules),
            "full_diagram_nodes": len(state.full_diagram.nodes),
            "full_diagram_edges": len(state.full_diagram.edges),
            "full_diagram_structural_score": state.full_validation.score,
            "full_diagram_structural_defects": len(state.full_validation.defects),
            "full_diagram_lanes": len({n.lane for n in state.full_diagram.nodes if n.lane}),
            "actors": list(state.decomposition.actors),
            "lane_mapping": lane_mapping,
            "parts": [
                {
                    "number": part.number,
                    "module": part.module_id,
                    "name": part.name,
                    "stem": part_files[part.number][0] if part.number in part_files else None,
                    "comes_from": part.comes_from,
                    "continues_to": part.continues_to,
                }
                for part in parts
            ],
            "plantuml_errors": compile_errors,
            "plantuml_checks": checks,
            "llm_calls": llm_calls,
            "total_execution_time_seconds": time.perf_counter() - started,
            "render": renders,
            "token_usage": self._token_usage(state, usage_tracker, usage_start),
            "run_output_dir": str(run_dir),
        }

        self._save(run_dir / "03_final_state.json", state)
        # A failed compilation leaves no .puml file rather than an empty one;
        # the error is in the metrics and the report.
        for name, text in (("overview", state.overview_plantuml), ("full", state.full_plantuml)):
            if text:
                (run_dir / f"{name}.puml").write_text(text, encoding="utf-8")
        (run_dir / "report.md").write_text(build_hierarchical_report(state), encoding="utf-8")
        (run_dir / "viewer.html").write_text(build_viewer(state, sample_id, {n: text for n, (_, text) in part_files.items()}), encoding="utf-8")
        self.log.info(
            "Hierarchical run finished: %d modules, coverage %.1f%%, outputs in %s",
            len(state.modules),
            100 * state.metrics["requirement_coverage"],
            run_dir,
        )
        return state

    # ============================================================
    # STAGES
    # ============================================================

    def _extract(
        self,
        chunks: list[DocumentChunk],
        title: str,
    ) -> tuple[list[tuple[DocumentChunk, list[Requirement]]], list[str]]:
        def extract_one(chunk: DocumentChunk) -> tuple[DocumentChunk, list[Requirement], str]:
            try:
                response = self.requirement_agent.run_chunk(chunk, document_title=title)
                return chunk, list(response.requirements), ""
            except Exception as exc:  # one bad chunk must not sink a 20-page run
                self.log.error("Requirement extraction failed for %s: %s", chunk.id, exc)
                return chunk, [], f"{chunk.id} ({chunk.section or 'no section'}): {exc}"

        def extract(chunk: DocumentChunk) -> list[tuple[DocumentChunk, list[Requirement], str]]:
            # The model sometimes skips whole sections of a long chunk; any
            # sizeable section left without a requirement is extracted again
            # on its own (duplicates are removed by consolidation).
            results = [extract_one(chunk)]
            if not results[0][2]:
                for part in missed_sections(self.chunker.sections(chunk), results[0][1]):
                    self.log.warning(
                        "No requirements extracted from section '%s' of %s; extracting it separately.",
                        part.section, chunk.id,
                    )
                    results.append(extract_one(part))
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = [item for items in pool.map(extract, chunks) for item in items]
        per_chunk = [(chunk, requirements) for chunk, requirements, _ in results]
        errors = [error for _, _, error in results if error]
        return per_chunk, errors

    def _decompose(
        self,
        requirements: list[Requirement],
        title: str,
    ) -> tuple[DecompositionPlan, bool]:
        limit = self.max_requirements_per_module
        if len(requirements) <= limit:
            module = ActivityModule(
                id="M1",
                name=title,
                objective="Complete system behaviour.",
                requirement_ids=[r.id for r in requirements],
            )
            plan = DecompositionPlan(system_name=title, modules=[module])
            return normalize_decomposition(plan, requirements, limit), False

        # ~25 tokens per compact requirement line.
        if 25 * len(requirements) > self.decomposition_token_budget:
            self.log.warning("Requirement set too large for one decomposition prompt; grouping by section.")
            return section_decomposition(requirements, limit, system_name=title), False

        try:
            plan = self.decomposition_agent.run(requirements, limit, document_title=title)
            return normalize_decomposition(plan, requirements, limit), True
        except Exception as exc:
            self.log.error("Decomposition agent failed (%s); grouping by section.", exc)
            return section_decomposition(requirements, limit, system_name=title), True

    def _run_module(
        self,
        module: ActivityModule,
        plan: DecompositionPlan,
        requirement_by_id: dict[str, Requirement],
        max_iterations: int,
        modules_dir: Path,
    ) -> ModuleResult:
        requirements = [requirement_by_id[rid] for rid in module.requirement_ids]
        text = module_requirement_text(module, plan, requirements)
        try:
            pipeline = self.pipeline_factory()
            result = pipeline.run(
                sample_id=module.id,
                requirement_text=text,
                max_iterations=max_iterations,
                output_dir=modules_dir,
                requirements=requirements,
                render=False,
            )
            if result.diagram is not None:
                result.diagram.title = module.name
            return ModuleResult(
                module=module,
                diagram=result.diagram,
                plantuml=result.final_plantuml,
                remaining_defects=result.defects,
                metrics={
                    "llm_calls": result.metrics.get("llm_calls", 0),
                    "best_iteration": result.metrics.get("best_iteration"),
                    "best_candidate_quality": result.metrics.get("best_candidate_quality"),
                    "repair_iterations": result.metrics.get("repair_iterations", 0),
                    "run_output_dir": result.metrics.get("run_output_dir"),
                    "token_usage": result.metrics.get("token_usage", {}).get("total"),
                },
            )
        except Exception as exc:
            self.log.error("Module %s (%s) failed: %s", module.id, module.name, exc)
            return ModuleResult(module=module, error=str(exc))

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _token_usage(state: HierarchicalState, tracker, start: int) -> dict[str, Any]:
        """Tokens and cost of the whole run, per agent, per module and for
        the document-level stages (extraction, decomposition, lane mapping)."""
        if tracker is None:
            return {}
        records = tracker.since(start)
        pricing = llm_usage.load_pricing()
        return {
            "total": llm_usage.summarize(records, pricing),
            "by_agent": llm_usage.group_by(records, "agent", pricing),
            "document_level": llm_usage.group_by([r for r in records if r["run"] is None], "agent", pricing),
            "by_module": {m.module.id: m.metrics.get("token_usage") for m in state.modules},
            "pricing_source": str(llm_usage.PRICING_FILE) if pricing else None,
        }

    def _harmonize_lanes(self, state: HierarchicalState) -> tuple[dict[str, str | None], bool]:
        """Map every module lane onto the decomposition's actors."""
        actors = state.decomposition.actors
        diagrams = [m.diagram for m in state.modules if m.diagram is not None]
        if not actors or not diagrams:
            return {}, False
        steps: dict[str, list[str]] = {}
        for diagram in diagrams:
            for node in diagram.nodes:
                if node.lane and node.lane.strip():
                    steps.setdefault(" ".join(node.lane.split()), []).append(node.label)
        mapping: dict[str, str | None] = {lane: match_actor(lane, actors) for lane in steps}
        unresolved = {lane: labels for lane, labels in steps.items() if mapping[lane] is None}
        used_llm = False
        if unresolved:
            used_llm = True
            try:
                mapping.update(self.lane_agent.run(unresolved, actors))
            except Exception as exc:
                self.log.warning("Lane mapping failed (%s); unmatched lanes follow the preceding step.", exc)
        for module in state.modules:
            if module.diagram is not None and apply_lanes(module.diagram, mapping):
                module.plantuml = self._compile(module.diagram, module.module.id, {}) or module.plantuml
        self.log.info("Swimlanes: %d lane names mapped onto %d actors.", len(steps), len(actors))
        return mapping, used_llm

    def _compile_checked(
        self,
        diagram,
        name: str,
        stem: Path,
        errors: dict[str, str],
        checks: dict[str, dict[str, Any]],
        links: dict[str, str] | None = None,
    ) -> str:
        """Compile, check with PlantUML, and scale down if it is too large to view."""
        text = self._compile(diagram, name, errors, links)
        if not text:
            return ""
        text, result, note = self._fit(text, self.syntax.check(text))
        return text if self._record_check(name, text, result, stem, errors, checks, note) else ""

    def _fit(self, text: str, result: PlantUMLCheck) -> tuple[str, PlantUMLCheck, str]:
        """Scale an oversized diagram down to PlantUML's default PNG limit while it stays readable."""
        if not (result.valid and result.oversized):
            return text, result, ""
        factor = math.floor(100 * 0.98 * DEFAULT_LIMIT_SIZE / max(result.width, result.height)) / 100
        if factor < MIN_FIT_SCALE:
            return text, result, ""
        scaled = text.replace("@startuml\n", f"@startuml\nscale {factor}\n", 1)
        check = self.syntax.check(scaled)
        if not check.valid or check.oversized:
            return text, result, ""
        return scaled, check, f"Scaled to {factor:.0%} (from {result.size}) to fit PlantUML's {DEFAULT_LIMIT_SIZE} px limit."

    def _record_check(
        self,
        name: str,
        text: str,
        result: PlantUMLCheck,
        stem: Path,
        errors: dict[str, str],
        checks: dict[str, dict[str, Any]],
        note: str = "",
    ) -> bool:
        checks[name] = {
            "valid": result.valid,
            "verified": result.verified,
            "message": result.message,
            "width": result.width,
            "height": result.height,
            "oversized": result.oversized,
            "note": note,
        }
        if result.oversized:
            self.log.warning(
                "%s diagram is %s; viewers that keep PlantUML's default %d px limit crop it. Open the SVG instead.",
                name, result.size, DEFAULT_LIMIT_SIZE,
            )
        if not result.verified:
            self.log.warning("%s diagram not checked by PlantUML: %s", name, result.message)
        if result.valid:
            return True
        self.log.error("PlantUML syntax check failed for %s diagram: %s", name, result.message)
        errors[name] = f"PlantUML syntax error: {result.message}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_name(stem.name + ".invalid.puml").write_text(text, encoding="utf-8")
        return False

    def _compile(self, diagram, name: str, errors: dict[str, str], links: dict[str, str] | None = None) -> str:
        try:
            return PlantUMLGenerator(links=links).render(diagram)
        except Exception as exc:
            self.log.error("PlantUML compilation failed for %s diagram (%s): %s", name, diagram.title, exc)
            errors[name] = str(exc)
            return ""

    def _render(self, plantuml_text: str, directory: Path, name: str) -> dict[str, Any]:
        if not plantuml_text:
            return {"rendered": False, "error": "empty PlantUML"}
        try:
            return self.renderer.render(plantuml_text, directory, name=name)
        except Exception as exc:
            return {"rendered": False, "error": str(exc)}

    @staticmethod
    def _save(path: Path, value: Any) -> None:
        if isinstance(value, list):
            payload = [v.model_dump(mode="json") if hasattr(v, "model_dump") else v for v in value]
        elif hasattr(value, "model_dump"):
            payload = value.model_dump(mode="json")
        else:
            payload = value
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


_STOPWORDS = {
    "the", "and", "for", "are", "that", "this", "with", "from", "its", "which",
    "into", "has", "have", "been", "will", "can", "all", "any", "not", "but",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def missed_sections(
    parts: list[DocumentChunk],
    requirements: list[Requirement],
    min_tokens: int = 80,
) -> list[DocumentChunk]:
    """Sections of a chunk that no extracted requirement comes from.

    Each requirement is attributed to the one section whose text its source
    sentence overlaps most, so a sentence that merely shares vocabulary with
    another section does not count for it.
    """
    if len(parts) < 2:
        return []
    part_words = [_words(part.text) for part in parts]
    hits = [0] * len(parts)
    for requirement in requirements:
        words = _words(requirement.source_sentence or requirement.text)
        if not words:
            continue
        scores = [len(words & pw) / len(words) for pw in part_words]
        best = max(range(len(parts)), key=scores.__getitem__)
        if scores[best] >= 0.5:
            hits[best] += 1
    return [
        part
        for part, count in zip(parts, hits)
        if count == 0 and part.estimated_tokens >= min_tokens
    ]


def module_requirement_text(
    module: ActivityModule,
    plan: DecompositionPlan,
    requirements: list[Requirement],
) -> str:
    """Build the bounded requirement text a module's agents work from."""
    module_ids = [m.id for m in plan.modules]
    names = {m.id: m.name for m in plan.modules}
    names.update({"START": "system start", "END": "system end"})
    incoming = [
        f"{names.get(t.source, t.source)}" + (f" [{t.guard}]" if t.guard else "")
        for t in plan.transitions
        if t.target == module.id
    ]
    outgoing = [
        f"{names.get(t.target, t.target)}" + (f" [{t.guard}]" if t.guard else "")
        for t in plan.transitions
        if t.source == module.id
    ]

    lines = [
        f"SYSTEM: {plan.system_name}",
    ]
    if plan.summary:
        lines.append(f"SYSTEM SUMMARY: {plan.summary}")
    lines += [
        f"MODULE {module.id} ({module_ids.index(module.id) + 1} of {len(module_ids)}): {module.name}",
        f"OBJECTIVE: {module.objective or module.name}",
        f"ENTERED FROM: {', '.join(incoming) or 'system start'}",
        f"CONTINUES TO: {', '.join(outgoing) or 'system end'}",
        "",
        "Model ONLY this module as a self-contained activity: begin where the",
        "module is entered and end with a final node where control passes on.",
        "A branch where the request is rejected, cancelled or fails, or the whole",
        "process stops, must end in its own final node whose label names that outcome",
        "(e.g. 'Request rejected', 'Peering cancelled'); any other final node means",
        "control passes on to the next module.",
        "",
    ]
    if plan.actors:
        lines += [*swimlane_instruction(plan.actors, module.lanes), ""]
    lines += [
        "MODELLING: when requirements come from use-case sections (Trigger, Preconditions,",
        "Stimulus/Response Sequences, Exceptions, Functional Requirements), the stimulus/response",
        "sequence is the main flow, step by step in its order. Model a decision only where the text",
        "names alternative outcomes that change the flow (an exception, a rejection, a trigger",
        "condition). Preconditions, postconditions and functional requirements that state a property",
        "or constraint rather than a step are traced to the actions they constrain (requirement_ids),",
        "not drawn as extra actions or decisions. Every requirement ID below must still appear in the",
        "requirement_ids of at least one node or edge.",
        "",
        "LABELS: actions are short verb phrases (at most 8 words, e.g. 'Send initiation request to",
        "Mediator') naming the step, with actors written out in full, no explanations. Decisions are",
        "short questions ('Sufficient resources acquired?'); guards are a few words ('yes', 'no',",
        "'new resource').",
        "",
        "SOURCE REQUIREMENTS:",
    ]
    current_section = None
    for requirement in requirements:
        if requirement.section != current_section:
            current_section = requirement.section
            lines.append(f"[{current_section or 'no section'}]")
        lines.append(f"- {requirement.id}: {requirement.source_sentence or requirement.text}")
    return "\n".join(lines)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:40] or "module"
