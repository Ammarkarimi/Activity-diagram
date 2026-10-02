from __future__ import annotations

import re
import threading

from src.agents.feedback_agent import FeedbackResult
from src.agents.requirement_agent import RequirementExtractionResponse
from src.document.chunker import DocumentChunker, estimate_tokens
from src.generation.plantuml_generator import PlantUMLGenerator
from src.models.domain import (
    ActivityDiagram,
    ActivityEdge,
    ActivityModule,
    ActivityNode,
    ActivityPlan,
    DecompositionPlan,
    DocumentChunk,
    ModuleTransition,
    NodeType,
    RepairResult,
    Requirement,
    RequirementMatrix,
    RequirementType,
    ReviewResult,
)
from src.pipeline.composer import build_overview_diagram, compose_full_diagram
from src.pipeline.decomposition import normalize_decomposition, section_decomposition
from src.pipeline.hierarchical import HierarchicalPipeline
from src.pipeline.orchestrator import MultiAgentPipeline
from src.pipeline.requirement_consolidation import consolidate_requirements
from src.validation.structural.structural_validator import StructuralValidator
from src.validation.syntax.plantuml_validator import PlantUMLSyntaxValidator


# ------------------------------------------------------------------
# Synthetic ~20 page specification
# ------------------------------------------------------------------

def make_long_spec(use_cases: int = 20, steps: int = 18) -> str:
    parts = ["# Online Retail Platform SRS", "", "## 1 Introduction", "",
             "This document specifies the behaviour of the platform. " * 20, ""]
    for u in range(1, use_cases + 1):
        parts += [f"## 3.{u} Use Case {u}", ""]
        parts += [f"### 3.{u}.1 Stimulus/Response Sequences", ""]
        for s in range(1, steps + 1):
            parts.append(
                f"The system performs step {s} of use case {u}, recording the outcome in the "
                f"audit log and notifying the operator about progress of workflow {u}."
            )
            if s % 4 == 0:
                parts.append("")
        parts.append("")
    return "\n".join(parts)


# ------------------------------------------------------------------
# Fake LLM that behaves like the structured-output client
# ------------------------------------------------------------------

class FakeLLM:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.max_user_chars: dict[str, int] = {}
        self._lock = threading.Lock()

    def complete(self, *, system: str, user: str, response_model):
        with self._lock:
            name = response_model.__name__
            self.calls[name] = self.calls.get(name, 0) + 1
            self.max_user_chars[name] = max(self.max_user_chars.get(name, 0), len(user))

        if response_model is RequirementExtractionResponse:
            chunk = user.split("=================\n", 1)[-1]
            sentences = re.findall(r"The system performs [^.]+\.", chunk)
            return RequirementExtractionResponse(
                requirements=[
                    Requirement(id=f"R{i}", text=s, type=RequirementType.ACTION, source_sentence=s,
                                dependencies=[f"R{i - 1}"] if i > 1 else [])
                    for i, s in enumerate(sentences, start=1)
                ],
                matrix=RequirementMatrix(),
            )
        if response_model is DecompositionPlan:
            ids = re.findall(r"^(R\d+) \[", user, re.MULTILINE)
            modules = [
                ActivityModule(id=f"M{k + 1}", name=f"Workflow {k + 1}", requirement_ids=ids[i:i + 18])
                for k, i in enumerate(range(0, len(ids), 18))
            ]
            return DecompositionPlan(system_name="Retail", modules=modules)
        if response_model is ActivityPlan:
            return ActivityPlan(objective="plan")
        if response_model is ActivityDiagram:
            reqs = re.findall(r"^- (R\d+): (.+)$", user, re.MULTILINE)
            nodes = [ActivityNode(id="N1", type=NodeType.INITIAL, label="Start")]
            for rid, text in reqs:
                nodes.append(ActivityNode(id=f"N{len(nodes) + 1}", type=NodeType.ACTION,
                                          label=text[:60], requirement_ids=[rid], lane="System"))
            nodes.append(ActivityNode(id=f"N{len(nodes) + 1}", type=NodeType.FINAL, label="Done"))
            edges = [
                ActivityEdge(id=f"E{i}", source=a.id, target=b.id)
                for i, (a, b) in enumerate(zip(nodes, nodes[1:]), start=1)
            ]
            return ActivityDiagram(title="Module", nodes=nodes, edges=edges)
        if response_model is ReviewResult:
            return ReviewResult(semantic_score=1.0)
        if response_model is FeedbackResult:
            return FeedbackResult()
        if response_model is RepairResult:
            return RepairResult(changed=False, diagram=ActivityDiagram())
        raise AssertionError(f"unexpected model {response_model}")


# ------------------------------------------------------------------
# Chunker
# ------------------------------------------------------------------

def test_chunker_respects_budget_and_tracks_sections():
    text = make_long_spec()
    assert estimate_tokens(text) > 10_000  # comparable to a ~20 page SRS
    chunks = DocumentChunker(max_tokens=800, overlap_tokens=100).chunk(text)

    assert len(chunks) > 10
    assert all(c.estimated_tokens <= 800 for c in chunks)
    assert chunks[0].context_before == ""
    assert all(c.context_before for c in chunks[1:])
    assert any("3.5 Use Case 5" in s for c in chunks for s in c.sections)
    # Nothing is lost: every behavioural sentence appears in some chunk.
    joined = "\n".join(c.text for c in chunks)
    assert joined.count("The system performs") == text.count("The system performs")


def test_chunker_splits_huge_paragraph_and_ignores_list_items_as_headings():
    paragraph = "The operator approves the order. " * 2000
    text = "## 2 Orders\n\n5. The system sends the confirmation to the customer\n\n" + paragraph
    chunks = DocumentChunker(max_tokens=500).chunk(text)
    assert len(chunks) > 10
    assert all(c.estimated_tokens <= 500 for c in chunks)
    assert all(c.section == "2 Orders" for c in chunks)


# ------------------------------------------------------------------
# Consolidation / decomposition
# ------------------------------------------------------------------

def test_consolidation_renumbers_remaps_and_drops_overlap_duplicates():
    c1 = DocumentChunk(id="C1", text="", section="A")
    c2 = DocumentChunk(id="C2", text="", section="B")
    r = lambda i, t, deps=(): Requirement(id=i, text=t, dependencies=list(deps))
    merged = consolidate_requirements([
        (c1, [r("R1", "User logs in"), r("R2", "System validates the password", ["R1"])]),
        (c2, [r("R1", "System validates the password"), r("R2", "System opens dashboard", ["R1"])]),
    ])
    assert [m.id for m in merged] == ["R1", "R2", "R3"]
    assert merged[1].dependencies == ["R1"]
    # dependency on the dropped duplicate points at the surviving copy
    assert merged[2].dependencies == ["R2"]
    assert merged[2].chunk_id == "C2" and merged[2].section == "B"


def test_normalize_decomposition_repairs_llm_output():
    reqs = [Requirement(id=f"R{i}", text=f"step {i}") for i in range(1, 13)]
    plan = DecompositionPlan(
        modules=[
            ActivityModule(id="M1", name="A", requirement_ids=["R1", "R2", "R99"]),
            ActivityModule(id="M2", name="B", requirement_ids=["R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]),
            ActivityModule(id="M3", name="Empty", requirement_ids=[]),
        ],
        transitions=[
            ModuleTransition(source="START", target="M1"),
            ModuleTransition(source="M1", target="M3"),
            ModuleTransition(source="M3", target="M2"),
            ModuleTransition(source="M2", target="GHOST"),
        ],
    )
    result = normalize_decomposition(plan, reqs, max_requirements_per_module=4)

    assigned = [rid for m in result.modules for rid in m.requirement_ids]
    assert sorted(assigned, key=lambda x: int(x[1:])) == [r.id for r in reqs]  # exactly once each
    assert all(len(m.requirement_ids) <= 4 for m in result.modules)
    assert [m.id for m in result.modules] == [f"M{i}" for i in range(1, len(result.modules) + 1)]
    ids = {m.id for m in result.modules} | {"START", "END"}
    assert all(t.source in ids and t.target in ids for t in result.transitions)
    assert any(t.target == "END" for t in result.transitions)


def test_section_decomposition_groups_by_section():
    reqs = [
        Requirement(id=f"R{i}", text="x", section=f"3 Features > 3.{1 + (i - 1) // 5} Use case")
        for i in range(1, 21)
    ]
    plan = section_decomposition(reqs, max_requirements_per_module=10)
    assert len(plan.modules) == 4
    assert plan.modules[0].requirement_ids == ["R1", "R2", "R3", "R4", "R5"]


# ------------------------------------------------------------------
# Composer
# ------------------------------------------------------------------

def _module_diagram(prefix_label: str, with_reject: bool = False) -> ActivityDiagram:
    nodes = [
        ActivityNode(id="N1", type=NodeType.INITIAL, label="Start"),
        ActivityNode(id="N2", type=NodeType.ACTION, label=f"{prefix_label} work", requirement_ids=["R1"]),
        ActivityNode(id="N3", type=NodeType.FINAL, label="Done"),
    ]
    edges = [ActivityEdge(id="E1", source="N1", target="N2")]
    if with_reject:
        nodes += [
            ActivityNode(id="N4", type=NodeType.DECISION, label="Accepted?"),
            ActivityNode(id="N5", type=NodeType.FINAL, label="Request rejected"),
        ]
        edges += [
            ActivityEdge(id="E2", source="N2", target="N4"),
            ActivityEdge(id="E3", source="N4", target="N3", guard="yes"),
            ActivityEdge(id="E4", source="N4", target="N5", guard="no"),
        ]
    else:
        edges.append(ActivityEdge(id="E2", source="N2", target="N3"))
    return ActivityDiagram(nodes=nodes, edges=edges)


def test_compose_full_diagram_inlines_modules_and_keeps_abnormal_ends():
    plan = DecompositionPlan(
        system_name="Sys",
        modules=[ActivityModule(id="M1", name="First", requirement_ids=["R1"]),
                 ActivityModule(id="M2", name="Second", requirement_ids=["R2"])],
        transitions=[ModuleTransition(source="START", target="M1"),
                     ModuleTransition(source="M1", target="M2"),
                     ModuleTransition(source="M2", target="END")],
    )
    overview = build_overview_diagram(plan)
    assert not StructuralValidator().validate(overview).defects
    full = compose_full_diagram(overview, {"M1": _module_diagram("A", with_reject=True), "M2": _module_diagram("B")})

    labels = {n.label for n in full.nodes}
    assert {"A work", "B work", "Request rejected", "End"} <= labels
    assert "Done" not in labels  # normal module ends continue to the next module
    assert len([n for n in full.nodes if n.type == NodeType.INITIAL]) == 1
    assert not [d for d in StructuralValidator().validate(full).defects if d.severity.value in {"HIGH", "CRITICAL"}]
    puml = PlantUMLGenerator().render(full)
    ok, message = PlantUMLSyntaxValidator()._regex_validate(puml)
    assert ok, message
    assert puml.index("A work") < puml.index("B work")


def test_overview_branches_become_decision():
    plan = normalize_decomposition(
        DecompositionPlan(
            modules=[ActivityModule(id="M1", name="Browse", requirement_ids=["R1"]),
                     ActivityModule(id="M2", name="Admin", requirement_ids=["R2"])],
            transitions=[ModuleTransition(source="START", target="M1", guard="customer"),
                         ModuleTransition(source="START", target="M2", guard="admin")],
        ),
        [Requirement(id="R1", text="a"), Requirement(id="R2", text="b")],
        10,
    )
    overview = build_overview_diagram(plan)
    assert any(n.type == NodeType.DECISION for n in overview.nodes)
    assert not StructuralValidator().validate(overview).defects
    PlantUMLGenerator().render(overview)


# ------------------------------------------------------------------
# End-to-end hierarchical run with the real agents and a fake LLM
# ------------------------------------------------------------------

def test_hierarchical_pipeline_end_to_end_on_long_document(tmp_path, monkeypatch):
    # Built-in syntax checks only: starting PlantUML for every iteration,
    # module and part makes this test take minutes.
    monkeypatch.setenv("PLANTUML_JAR", str(tmp_path / "no-plantuml.jar"))
    monkeypatch.setattr("src.generation.plantuml_tool.shutil.which", lambda *a, **k: None)
    llm = FakeLLM()
    text = make_long_spec()
    assert estimate_tokens(text) > 12_000
    pipeline = HierarchicalPipeline(
        llm=llm,
        chunk_tokens=900,
        max_requirements_per_module=20,
        max_workers=4,
        pipeline_factory=lambda: MultiAgentPipeline(llm=llm),
    )
    state = pipeline.run("retail", text, max_iterations=1, output_dir=tmp_path)

    assert len(state.chunks) > 5
    assert len(state.requirements) == 20 * 18
    assert len(state.modules) >= 18
    assert all(not m.error for m in state.modules)
    assert state.metrics["requirement_coverage"] == 1.0
    assert state.full_plantuml.startswith("@startuml")
    assert len(state.full_diagram.nodes) >= 20 * 18
    assert state.metrics["full_diagram_structural_defects"] == 0
    # Extraction and per-module prompts stay bounded although the document
    # is ~20 pages; only the decomposition prompt sees (compact) all
    # requirements.
    for name in ["RequirementExtractionResponse", "ActivityPlan", "ActivityDiagram", "ReviewResult"]:
        assert llm.max_user_chars[name] < 25_000 < len(text), name

    run_dir = tmp_path / next(p.name for p in tmp_path.iterdir())
    for name in ["00_chunks.json", "01_requirements.json", "02_decomposition.json",
                 "03_final_state.json", "overview.puml", "full.puml", "report.md"]:
        assert (run_dir / name).exists(), name
    assert "Requirement coverage | 100.0%" in (run_dir / "report.md").read_text()
