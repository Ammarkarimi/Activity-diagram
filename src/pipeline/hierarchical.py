from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.agents.decomposition_agent import DecompositionAgent
from src.agents.requirement_agent import RequirementAgent
from src.document.chunker import DocumentChunker, estimate_tokens
from src.generation.ir_sanitizer import IRSanitizer
from src.generation.plantuml_generator import PlantUMLGenerator
from src.generation.renderer import PlantUMLRenderer
from src.llm.openai_client import OpenAIClient
from src.models.domain import (
    ActivityModule,
    DecompositionPlan,
    DocumentChunk,
    HierarchicalState,
    ModuleResult,
    Requirement,
)
from src.pipeline.composer import build_overview_diagram, compose_full_diagram
from src.pipeline.decomposition import (
    normalize_decomposition,
    section_decomposition,
)
from src.pipeline.orchestrator import MultiAgentPipeline
from src.pipeline.requirement_consolidation import consolidate_requirements
from src.reporting.report import build_hierarchical_report
from src.validation.structural.structural_validator import StructuralValidator


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
        llm_calls += len(state.chunks)
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

        # ---------------- 5. overview + full diagram ----------------
        overview = build_overview_diagram(state.decomposition)
        state.overview_diagram = overview
        state.overview_plantuml = self._compile(overview)

        full = compose_full_diagram(
            overview,
            {m.module.id: m.diagram for m in state.modules},
            title=state.decomposition.system_name or title,
        )
        state.full_diagram = IRSanitizer.sanitize(full)
        state.full_validation = self.validator.validate(state.full_diagram)
        state.full_plantuml = self._compile(state.full_diagram)

        # ---------------- 6. coverage + report ----------------
        covered = {
            rid
            for m in state.modules
            if m.diagram is not None
            for element in [*m.diagram.nodes, *m.diagram.edges]
            for rid in element.requirement_ids
        }
        state.uncovered_requirement_ids = [r.id for r in state.requirements if r.id not in covered]

        renders = {
            "overview": self._render(state.overview_plantuml, run_dir, f"{sample_id}_overview"),
            "full": self._render(state.full_plantuml, run_dir, f"{sample_id}_full"),
        }
        for module in state.modules:
            if module.plantuml:
                renders[module.module.id] = self._render(
                    module.plantuml,
                    run_dir / "modules",
                    f"{module.module.id}_{_slug(module.module.name)}",
                )

        total = len(state.requirements)
        state.metrics = {
            "document_tokens": state.document_tokens,
            "chunk_count": len(state.chunks),
            "requirement_count": total,
            "module_count": len(state.modules),
            "failed_modules": [m.module.id for m in state.modules if m.error],
            "requirement_coverage": (total - len(state.uncovered_requirement_ids)) / total if total else 1.0,
            "remaining_module_defects": sum(len(m.remaining_defects) for m in state.modules),
            "full_diagram_nodes": len(state.full_diagram.nodes),
            "full_diagram_edges": len(state.full_diagram.edges),
            "full_diagram_structural_score": state.full_validation.score,
            "full_diagram_structural_defects": len(state.full_validation.defects),
            "llm_calls": llm_calls,
            "total_execution_time_seconds": time.perf_counter() - started,
            "render": renders,
            "run_output_dir": str(run_dir),
        }

        self._save(run_dir / "03_final_state.json", state)
        (run_dir / "overview.puml").write_text(state.overview_plantuml, encoding="utf-8")
        (run_dir / "full.puml").write_text(state.full_plantuml, encoding="utf-8")
        (run_dir / "report.md").write_text(build_hierarchical_report(state), encoding="utf-8")
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
        def extract(chunk: DocumentChunk) -> tuple[DocumentChunk, list[Requirement], str]:
            try:
                response = self.requirement_agent.run_chunk(chunk, document_title=title)
                return chunk, list(response.requirements), ""
            except Exception as exc:  # one bad chunk must not sink a 20-page run
                self.log.error("Requirement extraction failed for %s: %s", chunk.id, exc)
                return chunk, [], f"{chunk.id} ({chunk.section or 'no section'}): {exc}"

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(extract, chunks))
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
                },
            )
        except Exception as exc:
            self.log.error("Module %s (%s) failed: %s", module.id, module.name, exc)
            return ModuleResult(module=module, error=str(exc))

    # ============================================================
    # HELPERS
    # ============================================================

    def _compile(self, diagram) -> str:
        try:
            return PlantUMLGenerator().render(diagram)
        except Exception as exc:
            self.log.warning("PlantUML compilation failed for %s: %s", diagram.title, exc)
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
        "Use a final node whose label names the failure (e.g. 'Request",
        "rejected') only for flows that terminate the whole system.",
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
