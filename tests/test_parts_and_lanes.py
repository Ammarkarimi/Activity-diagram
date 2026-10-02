"""Shared swimlanes, linked part diagrams, the viewer, and the extraction guard."""
from src.document.chunker import DocumentChunker
from src.generation.plantuml_generator import PlantUMLGenerator
from src.models.domain import (
    ActivityDiagram,
    ActivityEdge,
    ActivityModule,
    ActivityNode,
    DecompositionPlan,
    DocumentChunk,
    HierarchicalState,
    ModuleTransition,
    NodeType,
    Requirement,
)
from src.pipeline.composer import build_overview_diagram, compose_full_diagram, split_into_parts
from src.pipeline.decomposition import normalize_decomposition
from src.pipeline.hierarchical import missed_sections
from src.pipeline.lanes import apply_lanes, canonical_actors, match_actor
from src.reporting.viewer import build_viewer

ACTORS = ["Mediator", "Local PA", "Service Registry", "Web Server", "Gateway", "DNS", "Peering CDNs", "Primary CDN"]


def test_lane_variants_match_their_actor():
    assert match_actor("mediator instance", ACTORS) == "Mediator"
    assert match_actor("Web servers", ACTORS) == "Web Server"
    assert match_actor("CDN gateway", ACTORS) == "Gateway"
    assert match_actor("gateways of participating CDNs", ACTORS) == "Gateway"
    assert match_actor("DNS", ACTORS) == "DNS"
    # Not actors, or ambiguous between actors: left for the lane agent.
    assert match_actor("functional policies", ACTORS) is None
    assert match_actor("CDNs", ACTORS) is None
    assert canonical_actors(["Mediator", "mediator", "Web Servers", "Web server"]) == ["Mediator", "Web Servers"]


def test_lane_that_is_not_an_actor_takes_the_preceding_actor():
    d = ActivityDiagram(
        nodes=[
            ActivityNode(id="N1", type=NodeType.INITIAL, label="Start"),
            ActivityNode(id="N2", type=NodeType.ACTION, label="Receive request", lane="mediator"),
            ActivityNode(id="N3", type=NodeType.ACTION, label="Deploy policies", lane="functional policies"),
            ActivityNode(id="N4", type=NodeType.FINAL, label="End"),
        ],
        edges=[
            ActivityEdge(id="E1", source="N1", target="N2"),
            ActivityEdge(id="E2", source="N2", target="N3"),
            ActivityEdge(id="E3", source="N3", target="N4"),
        ],
    )
    apply_lanes(d, {"mediator": "Mediator", "functional policies": None})
    assert [n.lane for n in d.nodes[1:3]] == ["Mediator", "Mediator"]


def test_decomposition_keeps_one_spelling_per_actor():
    requirements = [Requirement(id=f"R{i}", text=f"Step {i}") for i in range(1, 4)]
    plan = DecompositionPlan(
        actors=["Mediator", "mediator", "Local PA"],
        modules=[ActivityModule(id="M1", name="A", requirement_ids=["R1", "R2", "R3"],
                                lanes=["mediator instance", "PAs", "unknown thing"])],
    )
    plan = normalize_decomposition(plan, requirements, 25)
    assert plan.actors == ["Mediator", "Local PA"]
    assert plan.modules[0].lanes == ["Mediator", "Local PA"]


def test_section_skipped_by_extraction_is_detected():
    chunk = DocumentChunk(
        id="C2",
        text=(
            "2.2 Product Features\n\nThe registry stores resource descriptions for every provider.\n\n"
            "3.1 Service Registration\n\n" + "Web servers register new resources with the registry and "
            "increment the resource counter when a provider starts operating. " * 6
        ),
    )
    parts = DocumentChunker().sections(chunk)
    assert [p.section for p in parts] == ["2.2 Product Features", "3.1 Service Registration"]
    extracted = [Requirement(id="R1", text="x", source_sentence="The registry stores resource descriptions for every provider.")]
    assert [p.section for p in missed_sections(parts, extracted)] == ["3.1 Service Registration"]
    extracted.append(Requirement(id="R2", text="x", source_sentence="Web servers register new resources with the registry."))
    assert missed_sections(parts, extracted) == []


def _two_module_run():
    plan = DecompositionPlan(
        system_name="CDN",
        actors=["Web Server", "Mediator"],
        modules=[
            ActivityModule(id="M1", name="Trigger", requirement_ids=["R1"]),
            ActivityModule(id="M2", name="Negotiate", requirement_ids=["R2"]),
        ],
        transitions=[
            ModuleTransition(source="START", target="M1"),
            ModuleTransition(source="M1", target="M2"),
            ModuleTransition(source="M2", target="END"),
        ],
    )

    def module(label, lane, reject=False):
        nodes = [
            ActivityNode(id="N1", type=NodeType.INITIAL, label="Start"),
            ActivityNode(id="N2", type=NodeType.ACTION, label=label, lane=lane),
            ActivityNode(id="N3", type=NodeType.FINAL, label="Done"),
        ]
        edges = [ActivityEdge(id="E1", source="N1", target="N2"), ActivityEdge(id="E2", source="N2", target="N3")]
        if reject:
            nodes += [
                ActivityNode(id="N4", type=NodeType.DECISION, label="Overloaded?", lane=lane),
                ActivityNode(id="N5", type=NodeType.FINAL, label="Request rejected"),
            ]
            edges[1] = ActivityEdge(id="E2", source="N2", target="N4")
            edges += [
                ActivityEdge(id="E3", source="N4", target="N3", guard="yes"),
                ActivityEdge(id="E4", source="N4", target="N5", guard="no"),
            ]
        return ActivityDiagram(nodes=nodes, edges=edges)

    full = compose_full_diagram(
        build_overview_diagram(plan),
        {"M1": module("Detect hotspot", "Web Server", reject=True), "M2": module("Generate requirements", "Mediator")},
        title="CDN",
    )
    return plan, full


def test_parts_link_to_each_other_and_keep_lanes():
    plan, full = _two_module_run()
    parts = split_into_parts(full, plan)
    assert [(p.number, p.comes_from, p.continues_to) for p in parts] == [(1, [], [2]), (2, [1], [])]

    first = PlantUMLGenerator(links=parts[0].links).render(parts[0].diagram)
    second = PlantUMLGenerator(links=parts[1].links).render(parts[1].diagram)
    assert "|Web Server|" in first and "|Mediator|" not in first
    assert ":Continue in Part 2: Negotiate>" in first.splitlines()
    assert "stop" in first  # the rejection still ends the flow
    assert ":From Part 1: Trigger<" in second.splitlines()
    assert ":Generate requirements;" in second
    for text in (first, second):
        assert PlantUMLGenerator._blocks_balanced(text)
        lines = text.splitlines()
        assert not [i for i, line in enumerate(lines[:-1]) if line == "endif" and lines[i + 1] == "stop"]


def test_viewer_lists_every_diagram():
    plan, full = _two_module_run()
    parts = split_into_parts(full, plan)
    state = HierarchicalState(sample_id="cdn", decomposition=plan, full_plantuml="@startuml\n@enduml")
    state.metrics["parts"] = [
        {"number": p.number, "module": p.module_id, "name": p.name, "stem": f"part_{p.number:02d}",
         "comes_from": p.comes_from, "continues_to": p.continues_to}
        for p in parts
    ]
    page = build_viewer(state, "cdn", {1: "@startuml\n:</script>;\n@enduml"})
    assert '"svg": "cdn_full.svg"' in page
    assert '"svg": "parts/part_02.svg"' in page
    assert '"from": ["part-1"]' in page
    assert "</script>;" not in page  # embedded sources cannot close the script tag


def _refinement_pipeline():
    from src.generation.plantuml_generator import PlantUMLGenerator as Generator
    from src.pipeline.orchestrator import MultiAgentPipeline

    pipeline = MultiAgentPipeline.__new__(MultiAgentPipeline)
    pipeline.plantuml = Generator()
    return pipeline


def _linear(*labels_and_reqs):
    nodes = [ActivityNode(id="N0", type=NodeType.INITIAL, label="Start")]
    for i, (label, rids) in enumerate(labels_and_reqs, start=1):
        nodes.append(ActivityNode(id=f"N{i}", type=NodeType.ACTION, label=label, requirement_ids=rids))
    nodes.append(ActivityNode(id="NF", type=NodeType.FINAL, label="End"))
    ids = [n.id for n in nodes]
    return ActivityDiagram(nodes=nodes, edges=[
        ActivityEdge(id=f"E{i}", source=a, target=b) for i, (a, b) in enumerate(zip(ids, ids[1:]), start=1)
    ])


def test_refinement_that_loses_coverage_is_rejected():
    from src.models.domain import RequirementType

    pipeline = _refinement_pipeline()
    requirements = [
        Requirement(id="R1", text="Save buy instruction"),
        Requirement(id="R2", text="Save sell instruction"),
        Requirement(id="R3", text="Trading is fast", type=RequirementType.OTHER),
    ]
    draft = _linear(("Save buy instruction", ["R1"]), ("Save sell instruction", ["R2"]))
    merged = _linear(("Save buy / sell instruction", ["R1", "R2"]))
    dropped = _linear(("Save buy instruction", ["R1"]))
    assert pipeline._accept_refinement(draft, merged, requirements)[0]
    assert not pipeline._accept_refinement(draft, dropped, requirements)[0]
    # R3 is not behaviour: a diagram need not trace it.
    assert pipeline._calculate_completeness(requirements, pipeline._calculate_requirement_coverage(requirements, merged)) == 1.0
