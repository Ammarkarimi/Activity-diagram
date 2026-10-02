"""Meaning-based alignment of generated actions with gold-standard actions.

Label similarity only rewards shared words, so "Match stock ids" and
"Process instruction under specific matching rules" count as different
steps. Here an LLM judge pairs steps that describe the same behaviour,
one-to-one, whatever their wording. Report it next to the lexical score,
not instead of it.
"""
from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from src.models.domain import ActivityDiagram, NodeType


class StepPair(BaseModel):
    gold: int
    generated: int


class StepAlignment(BaseModel):
    pairs: list[StepPair] = Field(default_factory=list)


_SYSTEM = """
You compare two UML activity diagrams of the same specification: a GOLD
diagram drawn by an expert and a GENERATED diagram.

Pair each GOLD step with the GENERATED step that describes the same
behaviour: the same action with the same effect, even when worded
differently or at a slightly different level of detail ("Match stock ids"
and "Process instruction under matching rules" are the same step).

Rules:
- One-to-one: each gold step and each generated step appears in at most one
  pair. When several generated steps together make up one gold step, pair
  the one closest in meaning.
- Do not pair steps that merely share words but do different things
  ("Send request" and "Receive request" are different steps).
- Leave a step unpaired when nothing in the other diagram matches it.
Use the step numbers shown.
"""


def _actions(diagram: ActivityDiagram):
    return [n for n in diagram.nodes if n.type == NodeType.ACTION]


def semantic_mapping(
    gold: ActivityDiagram,
    generated: ActivityDiagram,
    llm,
    votes: int = 5,
) -> dict[str, tuple[str, float]]:
    """gold action ID -> (generated action ID, share of votes).

    A single judgement varies from call to call, so the judge is asked
    ``votes`` times and only pairs a majority agrees on are kept.
    """
    gold_actions, gen_actions = _actions(gold), _actions(generated)
    if not gold_actions or not gen_actions:
        return {}

    def listing(nodes):
        return "\n".join(
            f"{i}. {' '.join(n.label.split())}" + (f"  [{n.lane}]" if n.lane else "")
            for i, n in enumerate(nodes, start=1)
        )

    user = (
        f"GOLD STEPS\n{listing(gold_actions)}\n\n"
        f"GENERATED STEPS\n{listing(gen_actions)}\n\nReturn the pairs."
    )
    counts: Counter[tuple[int, int]] = Counter()
    for _ in range(votes):
        alignment = llm.complete(system=_SYSTEM, user=user, response_model=StepAlignment)
        seen = set()
        for pair in alignment.pairs:
            key = (pair.gold, pair.generated)
            if key not in seen and 1 <= pair.gold <= len(gold_actions) and 1 <= pair.generated <= len(gen_actions):
                seen.add(key)
                counts[key] += 1

    mapping: dict[str, tuple[str, float]] = {}
    used: set[str] = set()
    for (g_index, c_index), count in counts.most_common():
        if count * 2 <= votes:
            break
        g, c = gold_actions[g_index - 1].id, gen_actions[c_index - 1].id
        if g in mapping or c in used:
            continue
        mapping[g] = (c, count / votes)
        used.add(c)
    return mapping
