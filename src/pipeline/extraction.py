from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from src.agents.requirement_agent import RequirementAgent
from src.document.chunker import DocumentChunker
from src.models.domain import DocumentChunk, Requirement

_STOPWORDS = {
    "the", "and", "for", "are", "that", "this", "with", "from", "its", "which",
    "into", "has", "have", "been", "will", "can", "all", "any", "not", "but",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def missed_sections(
    parts: list[DocumentChunk],
    requirements: list[Requirement],
    min_tokens: int = 80,
) -> list[DocumentChunk]:
    """Sections of a chunk that no extracted requirement comes from.

    Each requirement is attributed to the one section whose text its source
    sentence overlaps most, so a sentence that merely shares vocabulary with
    another section does not count for it.
    """
    if len(parts) < 2:
        return []
    part_words = [_words(part.text) for part in parts]
    hits = [0] * len(parts)
    for requirement in requirements:
        words = _words(requirement.source_sentence or requirement.text)
        if not words:
            continue
        scores = [len(words & pw) / len(words) for pw in part_words]
        best = max(range(len(parts)), key=scores.__getitem__)
        if scores[best] >= 0.5:
            hits[best] += 1
    return [
        part
        for part, count in zip(parts, hits)
        if count == 0 and part.estimated_tokens >= min_tokens
    ]


def extract_chunked(
    agent: RequirementAgent,
    chunker: DocumentChunker,
    chunks: list[DocumentChunk],
    title: str,
    max_workers: int = 4,
    log: logging.Logger | None = None,
) -> tuple[list[tuple[DocumentChunk, list[Requirement]]], list[str]]:
    """Runs the Requirement Agent on every chunk, in parallel.

    The model sometimes skips whole sections of a long chunk; any sizeable
    section left without a requirement is extracted again on its own
    (duplicates are removed by consolidation).
    """
    log = log or logging.getLogger(__name__)

    def extract_one(chunk: DocumentChunk) -> tuple[DocumentChunk, list[Requirement], str]:
        try:
            response = agent.run_chunk(chunk, document_title=title)
            return chunk, list(response.requirements), ""
        except Exception as exc:  # one bad chunk must not sink the run
            log.error("Requirement extraction failed for %s: %s", chunk.id, exc)
            return chunk, [], f"{chunk.id} ({chunk.section or 'no section'}): {exc}"

    def extract(chunk: DocumentChunk) -> list[tuple[DocumentChunk, list[Requirement], str]]:
        results = [extract_one(chunk)]
        if not results[0][2]:
            for part in missed_sections(chunker.sections(chunk), results[0][1]):
                log.warning(
                    "No requirements extracted from section '%s' of %s; extracting it separately.",
                    part.section, chunk.id,
                )
                results.append(extract_one(part))
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = [item for items in pool.map(extract, chunks) for item in items]
    per_chunk = [(chunk, requirements) for chunk, requirements, _ in results]
    errors = [error for _, _, error in results if error]
    return per_chunk, errors
