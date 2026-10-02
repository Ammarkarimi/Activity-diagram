from __future__ import annotations

from collections import OrderedDict

from src.pipeline.lanes import canonical_actors, match_actor
from src.models.domain import (
    ActivityModule,
    DecompositionPlan,
    ModuleTransition,
    Requirement,
)

START = "START"
END = "END"


def _top_section(section: str) -> str:
    return section.split(" > ")[0].strip() if section else ""


def section_decomposition(
    requirements: list[Requirement],
    max_requirements_per_module: int,
    min_requirements_per_module: int = 3,
    system_name: str = "System",
) -> DecompositionPlan:
    """Deterministic fallback: group consecutive requirements by section.

    Used when the decomposition agent fails, or when the requirement list is
    too large for one decomposition prompt.
    """
    groups: "OrderedDict[str, list[Requirement]]" = OrderedDict()
    for requirement in requirements:
        key = _section_key(requirement.section)
        groups.setdefault(key, []).append(requirement)

    merged: list[tuple[str, list[Requirement]]] = []
    for name, items in groups.items():
        if merged and (
            len(merged[-1][1]) < min_requirements_per_module
            or len(items) < min_requirements_per_module
        ) and len(merged[-1][1]) + len(items) <= max_requirements_per_module:
            merged[-1] = (merged[-1][0] or name, merged[-1][1] + items)
        else:
            merged.append((name, list(items)))

    modules = [
        ActivityModule(
            id=f"M{index}",
            name=name or f"Part {index}",
            objective=f"Behaviour described in section '{name}'." if name else "",
            requirement_ids=[r.id for r in items],
        )
        for index, (name, items) in enumerate(merged, start=1)
    ]
    plan = DecompositionPlan(system_name=system_name, modules=modules, transitions=[])
    return normalize_decomposition(plan, requirements, max_requirements_per_module)


def _section_key(section: str) -> str:
    """Group by the second heading level when present (e.g. one use case)."""
    parts = [p.strip() for p in section.split(" > ")] if section else []
    return " > ".join(parts[:2])


def normalize_decomposition(
    plan: DecompositionPlan,
    requirements: list[Requirement],
    max_requirements_per_module: int,
) -> DecompositionPlan:
    """Make an (LLM-produced) decomposition safe to execute.

    Guarantees:
    - every requirement belongs to exactly one module;
    - no module is empty or larger than ``max_requirements_per_module``;
    - module IDs are M1..Mn;
    - transitions only reference existing modules / START / END, START has
      an outgoing transition, every module is reachable from START and
      can reach END.
    """
    order = {r.id: index for index, r in enumerate(requirements)}
    known = set(order)

    # ---------------- requirement assignment ----------------
    assigned: dict[str, str] = {}
    modules: list[ActivityModule] = []
    used_ids: set[str] = set()
    for module in plan.modules:
        module_id = (module.id or "").strip() or f"M{len(modules) + 1}"
        while module_id in used_ids or module_id in {START, END}:
            module_id = f"{module_id}_dup"
        used_ids.add(module_id)
        ids = []
        for rid in module.requirement_ids:
            if rid in known and rid not in assigned:
                assigned[rid] = module_id
                ids.append(rid)
        copy = module.model_copy(deep=True)
        copy.id = module_id
        copy.requirement_ids = ids
        modules.append(copy)

    if not modules:
        modules = [ActivityModule(id="M1", name=plan.system_name or "Main Flow", requirement_ids=[])]

    module_by_id = {m.id: m for m in modules}
    for requirement in requirements:
        if requirement.id in assigned:
            continue
        # Attach to the module owning the closest preceding requirement.
        target = None
        for previous in reversed(requirements[: order[requirement.id]]):
            if previous.id in assigned:
                target = module_by_id[assigned[previous.id]]
                break
        target = target or modules[0]
        target.requirement_ids.append(requirement.id)
        assigned[requirement.id] = target.id

    for module in modules:
        module.requirement_ids.sort(key=lambda rid: order[rid])

    # ---------------- remove empty modules ----------------
    removed = {m.id for m in modules if not m.requirement_ids}
    modules = [m for m in modules if m.requirement_ids]
    transitions = _bypass_removed(plan.transitions, removed)

    # ---------------- split oversized modules ----------------
    limit = max(1, max_requirements_per_module)
    split_modules: list[ActivityModule] = []
    for module in modules:
        if len(module.requirement_ids) <= limit:
            split_modules.append(module)
            continue
        # Balanced parts: 27 requirements with a limit of 25 become 14 + 13,
        # not 25 + 2.
        count = -(-len(module.requirement_ids) // limit)
        size = -(-len(module.requirement_ids) // count)
        parts = [
            module.requirement_ids[i:i + size]
            for i in range(0, len(module.requirement_ids), size)
        ]
        part_ids = [f"{module.id}__p{k}" for k in range(1, len(parts) + 1)]
        for k, (part_id, ids) in enumerate(zip(part_ids, parts), start=1):
            split_modules.append(
                ActivityModule(
                    id=part_id,
                    name=f"{module.name} (part {k}/{len(parts)})",
                    objective=module.objective,
                    requirement_ids=ids,
                    lanes=list(module.lanes),
                )
            )
        # Incoming edges go to the first part, outgoing leave the last part.
        new_transitions = []
        for t in transitions:
            source = part_ids[-1] if t.source == module.id else t.source
            target = part_ids[0] if t.target == module.id else t.target
            new_transitions.append(ModuleTransition(source=source, target=target, guard=t.guard))
        for a, b in zip(part_ids, part_ids[1:]):
            new_transitions.append(ModuleTransition(source=a, target=b))
        transitions = new_transitions
    modules = split_modules

    # ---------------- renumber ----------------
    rename = {m.id: f"M{index}" for index, m in enumerate(modules, start=1)}
    for module in modules:
        module.id = rename[module.id]
    transitions = [
        ModuleTransition(
            source=rename.get(t.source, t.source),
            target=rename.get(t.target, t.target),
            guard=(t.guard or "").strip() or None,
        )
        for t in transitions
    ]

    # ---------------- repair transitions ----------------
    valid = {m.id for m in modules}
    seen: set[tuple[str, str]] = set()
    clean: list[ModuleTransition] = []
    for t in transitions:
        if t.source not in valid | {START} or t.target not in valid | {END}:
            continue
        if t.source == START and t.target == END:
            continue
        if t.source == t.target or (t.source, t.target) in seen:
            continue
        seen.add((t.source, t.target))
        clean.append(t)

    if not clean:
        ids = [START] + [m.id for m in modules] + [END]
        clean = [ModuleTransition(source=a, target=b) for a, b in zip(ids, ids[1:])]

    ordered_ids = [m.id for m in modules]
    if not any(t.source == START for t in clean):
        clean.insert(0, ModuleTransition(source=START, target=ordered_ids[0]))

    # Every module reachable from START: attach orphans after the previous module.
    reachable = _reachable(clean, START)
    for index, module_id in enumerate(ordered_ids):
        if module_id in reachable:
            continue
        previous = ordered_ids[index - 1] if index > 0 else START
        clean.append(ModuleTransition(source=previous, target=module_id))
        reachable = _reachable(clean, START)

    # Every module can terminate: modules without successors go to END.
    for module_id in ordered_ids:
        if not any(t.source == module_id for t in clean):
            clean.append(ModuleTransition(source=module_id, target=END))
    if not any(t.target == END for t in clean):
        clean.append(ModuleTransition(source=ordered_ids[-1], target=END))

    # One spelling per actor; module lanes refer to those actors.
    actors = canonical_actors(plan.actors or [lane for m in modules for lane in m.lanes])
    for module in modules:
        module.lanes = list(dict.fromkeys(
            actor for actor in (match_actor(lane, actors) for lane in module.lanes) if actor
        ))

    return DecompositionPlan(
        system_name=plan.system_name or "System",
        summary=plan.summary,
        actors=actors,
        modules=modules,
        transitions=clean,
    )


def _bypass_removed(transitions: list[ModuleTransition], removed: set[str]) -> list[ModuleTransition]:
    if not removed:
        return list(transitions)
    result = [t for t in transitions if t.source not in removed and t.target not in removed]
    for module_id in removed:
        incoming = [t for t in transitions if t.target == module_id and t.source not in removed]
        outgoing = [t for t in transitions if t.source == module_id and t.target not in removed]
        for i in incoming:
            for o in outgoing:
                result.append(ModuleTransition(source=i.source, target=o.target, guard=i.guard or o.guard))
    return result


def _reachable(transitions: list[ModuleTransition], start: str) -> set[str]:
    seen = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for t in transitions:
            if t.source == node and t.target not in seen:
                seen.add(t.target)
                frontier.append(t.target)
    return seen
