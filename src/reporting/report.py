from __future__ import annotations

from collections import Counter

from src.models.domain import HierarchicalState


def build_hierarchical_report(state: HierarchicalState) -> str:
    """Human-readable Markdown summary of a hierarchical run."""
    m = state.metrics
    plan = state.decomposition
    lines = [
        f"# Activity Diagram Report: {state.sample_id}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Document size (est. tokens) | {state.document_tokens} |",
        f"| Chunks | {len(state.chunks)} |",
        f"| Requirements extracted | {len(state.requirements)} |",
        f"| Modules (sub-activities) | {len(state.modules)} |",
        f"| Requirement coverage | {100 * m.get('requirement_coverage', 0):.1f}% |",
        f"| Remaining module defects | {m.get('remaining_module_defects', 0)} |",
        f"| Full diagram nodes / edges | {m.get('full_diagram_nodes', 0)} / {m.get('full_diagram_edges', 0)} |",
        f"| Full diagram structural score | {m.get('full_diagram_structural_score', 0):.2f} |",
        f"| LLM calls | {m.get('llm_calls', 0)} |",
        f"| Time (s) | {m.get('total_execution_time_seconds', 0):.1f} |",
        "",
        "Outputs: `overview.puml` (one action per module), `full.puml` (all",
        "modules inlined), and `modules/` (one diagram per module, with each",
        "agent's intermediate output).",
        "",
    ]

    if plan is not None:
        names = {mod.id: mod.name for mod in plan.modules}
        names.update({"START": "START", "END": "END"})
        lines += ["## Module flow", ""]
        for t in plan.transitions:
            guard = f" [{t.guard}]" if t.guard else ""
            lines.append(f"- {names.get(t.source, t.source)} -> {names.get(t.target, t.target)}{guard}")
        lines.append("")

    lines += [
        "## Modules",
        "",
        "| ID | Module | Requirements | Nodes | Remaining defects | Best iteration | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in state.modules:
        nodes = len(result.diagram.nodes) if result.diagram else 0
        status = f"FAILED: {result.error[:80]}" if result.error else "ok"
        lines.append(
            f"| {result.module.id} | {result.module.name} | {len(result.module.requirement_ids)} | "
            f"{nodes} | {len(result.remaining_defects)} | {result.metrics.get('best_iteration', '-')} | {status} |"
        )
    lines.append("")

    defects = [d for r in state.modules for d in r.remaining_defects]
    if defects:
        lines += ["## Remaining defects by category", ""]
        for category, count in Counter(d.category for d in defects).most_common():
            lines.append(f"- {category}: {count}")
        lines.append("")

    if state.full_validation and state.full_validation.defects:
        lines += ["## Full-diagram structural findings", ""]
        for defect in state.full_validation.defects[:30]:
            lines.append(f"- [{defect.severity.value}] {defect.category}: {defect.description}")
        lines.append("")

    if state.uncovered_requirement_ids:
        by_id = {r.id: r for r in state.requirements}
        lines += ["## Requirements not traced to any diagram element", ""]
        for rid in state.uncovered_requirement_ids:
            lines.append(f"- {rid}: {by_id[rid].text}")
        lines.append("")

    if state.extraction_errors:
        lines += ["## Extraction errors", ""]
        lines += [f"- {error}" for error in state.extraction_errors]
        lines.append("")

    return "\n".join(lines)
