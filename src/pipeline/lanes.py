"""One set of swimlanes (actors) for every module of a hierarchical run.

Module diagrams are generated independently, so without a shared actor list
each one invents its own lanes, often from noun phrases that are not actors
at all ("functional policies", "load distribution"). The full diagram then
has dozens of lanes. These helpers map every lane onto the actors chosen by
the decomposition.
"""
from __future__ import annotations

import re

import networkx as nx

from src.models.domain import ActivityDiagram

_IGNORED = {"the", "a", "an", "of", "instance", "instances"}


def lane_key(name: str) -> tuple[str, ...]:
    """Words of a lane name, lower-cased and singular ("Web Servers" == "web server")."""
    words = []
    for word in re.findall(r"[A-Za-z0-9]+", name or ""):
        if re.fullmatch(r"[A-Z]{2,}s", word):
            word = word[:-1]  # "PAs", "CDNs"
        word = word.lower()
        if word in _IGNORED:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return tuple(words)


def canonical_actors(names: list[str]) -> list[str]:
    """Distinct actor names, first spelling wins."""
    seen: dict[tuple[str, ...], str] = {}
    for name in names:
        name = " ".join((name or "").split())
        key = lane_key(name)
        if key and key not in seen:
            seen[key] = name
    return list(seen.values())


def match_actor(lane: str, actors: list[str]) -> str | None:
    """The actor a lane name refers to, or None when it is unclear.

    Matches the same words, or one name's words contained in the other's
    ("CDN gateway" -> "Gateway", "PAs" -> "Local PA") when exactly one actor
    matches best.
    """
    key = lane_key(lane)
    if not key:
        return None
    keys = {actor: lane_key(actor) for actor in actors}
    for actor, actor_key in keys.items():
        if actor_key == key:
            return actor
    words = set(key)
    candidates = []
    for actor, actor_key in keys.items():
        actor_words = set(actor_key)
        if actor_words and (actor_words <= words or words <= actor_words):
            candidates.append((len(actor_words & words), actor))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def apply_lanes(diagram: ActivityDiagram, mapping: dict[str, str | None]) -> int:
    """Rename lanes with ``mapping``; returns how many nodes changed.

    A lane mapped to None is not an actor: its nodes take the lane of the
    nearest preceding node that has one (the actor already in control).
    """
    changed = 0
    unresolved = []
    for node in diagram.nodes:
        if not node.lane:
            continue
        lane = " ".join(node.lane.split())
        target = mapping.get(lane, lane)
        if target is None:
            unresolved.append(node)
            continue
        if target != node.lane:
            node.lane = target
            changed += 1
    if not unresolved:
        return changed

    graph = nx.DiGraph()
    graph.add_nodes_from(n.id for n in diagram.nodes)
    graph.add_edges_from((e.source, e.target) for e in diagram.edges)
    resolved = {n.id: n.lane for n in diagram.nodes if n not in unresolved and n.lane}
    fallback = next(iter(resolved.values()), None)
    for node in unresolved:
        lane = None
        # Breadth-first over predecessors: the closest actor wins.
        for _, predecessor in nx.bfs_edges(graph.reverse(copy=False), node.id):
            if predecessor in resolved:
                lane = resolved[predecessor]
                break
        node.lane = lane or fallback
        changed += 1
    return changed


def swimlane_instruction(actors: list[str], main: list[str] | None = None) -> list[str]:
    """Prompt lines telling the planning agents which lanes to use."""
    main = main or actors
    others = [a for a in actors if a not in main]
    return [
        "SWIMLANES: put every action and decision in the lane of the actor that performs it,",
        f"using exactly these names: {', '.join(main)}."
        + (f" Other actors of the system, only if they act here: {', '.join(others)}." if others else ""),
        "Do not create lanes for data, policies, protocols, capabilities or groups of steps.",
    ]
