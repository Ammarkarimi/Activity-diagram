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
        "Outputs: `viewer.html` (open in a browser: every diagram with zoom and",
        "scrolling), `overview.puml` (one action per module), `full.puml` (all",
        "modules inlined), `parts/` (the full diagram cut into one linked part",
        "per module) and `modules/` (each module's agent outputs).",
        "",
    ]

    actors = m.get("actors") or []
    if actors:
        mapping = m.get("lane_mapping") or {}
        renamed = {lane: actor for lane, actor in mapping.items() if lane != actor}
        lines += [
            "## Swimlanes",
            "",
            f"Actors ({len(actors)}): {', '.join(actors)}. "
            f"The full diagram uses {m.get('full_diagram_lanes', 0)} lanes.",
            "",
        ]
        if renamed:
            lines += ["Lane names from module diagrams mapped onto these actors:", ""]
            lines += [
                f"- {lane} -> {actor or '(not an actor: lane of the preceding step)'}"
                for lane, actor in sorted(renamed.items())
            ]
            lines.append("")

    parts = m.get("parts") or []
    if parts:
        lines += ["## Parts", "", "| Part | Module | From | Continues in | File |", "|---|---|---|---|---|"]
        for part in parts:
            comes = ", ".join(f"Part {n}" for n in part["comes_from"]) or "start"
            goes = ", ".join(f"Part {n}" for n in part["continues_to"]) or "end"
            file = f"`parts/{part['stem']}.puml`" if part.get("stem") else "not written"
            lines.append(f"| {part['number']}: {part['name']} | {part['module']} | {comes} | {goes} | {file} |")
        lines.append("")

    compile_errors = m.get("plantuml_errors") or {}
    if compile_errors:
        lines += ["## PlantUML errors", ""]
        lines += [f"- `{name}` was not written: {error}" for name, error in compile_errors.items()]
        lines.append("")

    checks = m.get("plantuml_checks") or {}
    if checks:
        lines += [
            "## PlantUML checks",
            "",
            "| Diagram | Syntax | Size | Notes |",
            "|---|---|---|---|",
        ]
        for name, check in checks.items():
            if not check["valid"]:
                syntax = f"ERROR: {check['message']}"
            elif check["verified"]:
                syntax = "valid"
            else:
                syntax = "not verified: PlantUML not installed"
            size = f"{check['width']} x {check['height']} px" if check.get("width") else "-"
            notes = [check["note"]] if check.get("note") else []
            if check.get("oversized"):
                notes.append(
                    "Larger than PlantUML's default 4096 px PNG limit, so editor previews and the "
                    "PlantUML server crop it: open the SVG."
                )
            cell = " ".join(notes).replace("|", "/")
            lines.append(f"| {name} | {syntax.replace('|', '/')} | {size} | {cell} |")
        lines.append("")

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
