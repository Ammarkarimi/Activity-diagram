from __future__ import annotations

import re

from src.models.domain import DocumentChunk, Requirement

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _normalized(text: str) -> str:
    return " ".join(_WORD.findall((text or "").lower()))


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def consolidate_requirements(
    per_chunk: list[tuple[DocumentChunk, list[Requirement]]],
    duplicate_threshold: float = 0.9,
    window: int = 40,
) -> list[Requirement]:
    """Merge chunk-local requirement lists into one global R1..Rn list.

    - IDs are renumbered in document order.
    - Chunk-local dependencies are remapped to the new global IDs.
    - Near-duplicates (usually caused by chunk overlap) are dropped; any
      dependency on a dropped duplicate is redirected to the surviving copy.
      Only the last ``window`` requirements are compared, which keeps this
      linear for very long documents while still catching overlap duplicates.
    - Section and chunk provenance is recorded on every requirement.
    """
    result: list[Requirement] = []

    for chunk, requirements in per_chunk:
        local_to_global: dict[str, str] = {}
        pending: list[tuple[Requirement, list[str]]] = []

        for requirement in requirements:
            duplicate = _find_duplicate(requirement, result, duplicate_threshold, window)
            if duplicate is not None:
                local_to_global[requirement.id] = duplicate.id
                continue

            global_id = f"R{len(result) + 1}"
            local_to_global[requirement.id] = global_id
            copy = requirement.model_copy(deep=True)
            copy.id = global_id
            copy.chunk_id = chunk.id
            copy.section = requirement.section or chunk.section
            copy.source_sentence = requirement.source_sentence or requirement.text
            result.append(copy)
            pending.append((copy, list(requirement.dependencies)))

        for copy, dependencies in pending:
            remapped: list[str] = []
            for dependency in dependencies:
                target = local_to_global.get(dependency)
                if target and target != copy.id and target not in remapped:
                    remapped.append(target)
            copy.dependencies = remapped

    return result


def _find_duplicate(
    requirement: Requirement,
    existing: list[Requirement],
    threshold: float,
    window: int,
) -> Requirement | None:
    """A duplicate is the same statement extracted twice (chunk overlap).

    Either the normalized text is identical, or both come from the same
    source sentence with near-identical wording. Requirements that differ
    in a detail ("step 1" vs "step 2") are never merged.
    """
    text = _normalized(requirement.text)
    source = _normalized(requirement.source_sentence)
    for candidate in reversed(existing[-window:]):
        if candidate.type != requirement.type:
            continue
        if _normalized(candidate.text) == text:
            return candidate
        if (
            source
            and _normalized(candidate.source_sentence) == source
            and _similarity(candidate.text, requirement.text) >= threshold
        ):
            return candidate
    return None


def compact_requirement_lines(requirements: list[Requirement], max_chars: int = 220) -> str:
    """One line per requirement, grouped under section headings.

    Used where the full model dump would be too large (e.g. decomposing a
    20-page specification).
    """
    lines = []
    current_section = None
    for requirement in requirements:
        if requirement.section != current_section:
            current_section = requirement.section
            lines.append(f"## {current_section or '(no section)'}")
        text = " ".join(requirement.text.split())
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        lines.append(f"{requirement.id} [{requirement.type.value}]: {text}")
    return "\n".join(lines)
