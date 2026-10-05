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


# Head nouns of things that do not act: media, groupings, artefacts and
# operations. "Local Network", "Multicast Group" and "Encryption/Decryption"
# were chosen as lanes although the prompt rules them out.
_NOT_ACTORS = {
    "network", "group", "address", "channel", "protocol", "message", "data",
    "information", "policy", "requirement", "encryption", "decryption",
    "communication", "connection", "session", "mode", "capability", "function",
}


def is_actor_name(name: str) -> bool:
    """False for lane names whose head noun is not something that acts."""
    words = re.findall(r"[A-Za-z]+", name or "")
    if not words:
        return False
    head = lane_key(words[-1])
    return not head or head[0] not in _NOT_ACTORS


def canonical_actors(names: list[str]) -> list[str]:
    """Distinct actor names, first spelling wins; non-actors are dropped."""
    seen: dict[tuple[str, ...], str] = {}
    for name in names:
        # "Peering Agent (PA)" is "Peering Agent": the abbreviation is not a word of the name.
        name = re.sub(r"\s*\([A-Z]{2,6}s?\)", "", name or "")
        name = " ".join(name.split())
        key = lane_key(name)
        if key and key not in seen and is_actor_name(name):
            seen[key] = name
    # A generic name next to its specific roles ("Agent" with "Sending Agent"
    # and "Receiving Agent") is not a participant of its own.
    return [
        name for key, name in seen.items()
        if not any(set(key) < set(other) for other in seen if other != key)
    ]


_DEFINITION = re.compile(r"((?:[A-Z][\w-]*\s+){0,5}[A-Z][\w-]*)\s*\(([A-Z]{2,6})\)")


def spec_abbreviations(text: str) -> dict[str, str]:
    """Abbreviations the specification defines: "Peering Agent (PA)" -> {"PA": "Peering Agent"}.

    The full name is the run of capitalised words before the parenthesis
    whose initials spell the abbreviation.
    """
    found: dict[str, str] = {}
    for words, abbr in _DEFINITION.findall(text or ""):
        parts = words.split()
        if len(parts) < len(abbr):
            continue
        name = parts[-len(abbr):]
        if "".join(w[0] for w in name).upper() == abbr:
            found.setdefault(abbr, " ".join(name))
    return found


_QUALIFIERS = ("local", "external", "remote", "peer", "primary", "foreign")


def spec_qualified_actors(actors: list[str], text: str, abbreviations: dict[str, str] | None = None) -> list[str]:
    """Use the qualifier the specification itself gives an actor most often.

    The specification follows one instance of a participant ("its local PA")
    and mentions the others in passing ("interacts with external PAs"), so
    "External Peering Agent" becomes "Local Peering Agent" when the text says
    "local PA" more often. Unqualified names are kept.
    """
    expanded = expand_abbreviations(text or "", abbreviations).casefold()
    expanded = " ".join(expanded.split())
    result = []
    for actor in actors:
        words = actor.split()
        if len(words) < 2 or words[0].casefold() not in _QUALIFIERS:
            result.append(actor)
            continue
        base = " ".join(words[1:])
        counts = {
            q: len(re.findall(rf"\b{q} {re.escape(base.casefold())}", expanded))
            for q in _QUALIFIERS
        }
        best = max(counts, key=counts.get)
        if counts[best] > counts[words[0].casefold()]:
            result.append(f"{best.capitalize()} {base}")
        else:
            result.append(actor)
    return canonical_actors(result)


def expand_abbreviations(name: str, abbreviations: dict[str, str] | None) -> str:
    """Write defined abbreviations out: "Local PA" -> "Local Peering Agent"."""
    if not abbreviations:
        return name
    def full(match: re.Match) -> str:
        word = match.group(0)
        base = word[:-1] if word.endswith("s") and word[:-1] in abbreviations else word
        return abbreviations.get(base, word)
    return re.sub(r"\b[A-Z]{2,6}s?\b", full, name or "")


def match_actor(lane: str, actors: list[str], abbreviations: dict[str, str] | None = None) -> str | None:
    """The actor a lane name refers to, or None when it is unclear.

    Matches the same words, or one name's words contained in the other's
    ("CDN gateway" -> "Gateway", "PAs" -> "Local PA") when exactly one actor
    matches best. Abbreviations the specification defines are written out
    first, so "Local PA" matches "Peering Agent", not "Policy Agent".
    """
    key = lane_key(expand_abbreviations(lane, abbreviations))
    if not key:
        return None
    keys = {actor: lane_key(expand_abbreviations(actor, abbreviations)) for actor in actors}
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
