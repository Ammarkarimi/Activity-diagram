from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models.domain import DocumentChunk


def estimate_tokens(text: str) -> int:
    """Cheap, provider-independent token estimate (~4 characters/token)."""
    return max(0, (len(text or "") + 3) // 4)


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(r"^((?:\d+\.)*\d+)\.?\s+([A-Z][^\n]{0,100})$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])")


@dataclass
class _Block:
    text: str
    start: int
    section: str
    is_heading: bool = False


@dataclass
class _ChunkBuilder:
    blocks: list[_Block] = field(default_factory=list)
    tokens: int = 0


class DocumentChunker:
    """Section-aware splitter for long requirement specifications.

    The document is split on headings and paragraphs, then paragraphs are
    packed into chunks of at most ``max_tokens``. Paragraphs that are larger
    than a chunk are split on sentence boundaries. Every chunk carries the
    heading path it belongs to, plus the tail of the previous chunk as
    read-only context, so extraction agents never see a requirement without
    its surrounding section.
    """

    def __init__(self, max_tokens: int = 2500, overlap_tokens: int = 150) -> None:
        if max_tokens < 100:
            raise ValueError("max_tokens must be at least 100.")
        self.max_tokens = max_tokens
        self.overlap_tokens = max(0, min(overlap_tokens, max_tokens // 2))

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def chunk(self, text: str) -> list[DocumentChunk]:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        blocks = self._split_blocks(text)
        if not blocks:
            return []

        builders: list[_ChunkBuilder] = []
        current = _ChunkBuilder()

        for block in self._explode_large_blocks(blocks):
            block_tokens = estimate_tokens(block.text)
            # Never leave a heading dangling at the end of a chunk.
            if current.blocks and current.tokens + block_tokens + 1 > self.max_tokens:
                carried: list[_Block] = []
                while current.blocks and current.blocks[-1].is_heading:
                    carried.insert(0, current.blocks.pop())
                if current.blocks:
                    builders.append(current)
                    current = _ChunkBuilder()
                for heading in carried:
                    current.blocks.append(heading)
                    current.tokens += estimate_tokens(heading.text)
            current.blocks.append(block)
            current.tokens += block_tokens + 1  # +1 for the paragraph separator

        if current.blocks:
            builders.append(current)

        chunks: list[DocumentChunk] = []
        previous_text = ""
        for index, builder in enumerate(builders, start=1):
            chunk_text = "\n\n".join(b.text for b in builder.blocks).strip()
            sections: list[str] = []
            for b in builder.blocks:
                if b.section and b.section not in sections:
                    sections.append(b.section)
            first = builder.blocks[0]
            last = builder.blocks[-1]
            chunks.append(
                DocumentChunk(
                    id=f"C{index}",
                    text=chunk_text,
                    section=first.section,
                    sections=sections,
                    start_char=first.start,
                    end_char=last.start + len(last.text),
                    context_before=self._tail(previous_text),
                    estimated_tokens=estimate_tokens(chunk_text),
                )
            )
            previous_text = chunk_text
        return chunks

    # ------------------------------------------------------------
    # Block splitting
    # ------------------------------------------------------------

    def _split_blocks(self, text: str) -> list[_Block]:
        blocks: list[_Block] = []
        heading_stack: list[tuple[int, str]] = []
        paragraph: list[str] = []
        paragraph_start = 0
        offset = 0

        def section_path() -> str:
            return " > ".join(title for _, title in heading_stack)

        def flush() -> None:
            nonlocal paragraph
            body = "\n".join(paragraph).strip()
            if body:
                blocks.append(_Block(body, paragraph_start, section_path()))
            paragraph = []

        for line in text.split("\n"):
            line_start = offset
            offset += len(line) + 1
            stripped = line.strip()

            if not stripped:
                flush()
                continue

            heading = self._heading(stripped)
            if heading is not None:
                flush()
                level, title = heading
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                blocks.append(_Block(stripped, line_start, section_path(), is_heading=True))
                continue

            if not paragraph:
                paragraph_start = line_start
            paragraph.append(line.rstrip())

        flush()
        return blocks

    @staticmethod
    def _heading(line: str) -> tuple[int, str] | None:
        match = _MARKDOWN_HEADING.match(line)
        if match:
            return len(match.group(1)), " ".join(match.group(2).split())

        if len(line) > 100 or line.endswith((".", ",", ";", ":")):
            return None

        match = _NUMBERED_HEADING.match(line)
        if match:
            number, title = match.group(1), match.group(2)
            words = len(title.split())
            # "5. The system sends a reply" is a list item; "5. Overview" or
            # "3.1.2 Trigger" are headings.
            single_level_list_item = "." not in number and line[len(number):].startswith(".") and words > 5
            if words <= 12 and not single_level_list_item:
                return number.count(".") + 1, " ".join(line.split())

        letters = [ch for ch in line if ch.isalpha()]
        if len(letters) >= 4 and all(ch.isupper() for ch in letters) and len(line.split()) <= 10:
            return 1, " ".join(line.split())

        return None

    def _explode_large_blocks(self, blocks: list[_Block]) -> list[_Block]:
        # Leave headroom for headings carried over to the next chunk.
        budget = int(self.max_tokens * 0.8)
        result: list[_Block] = []
        for block in blocks:
            if estimate_tokens(block.text) <= budget:
                result.append(block)
                continue
            piece: list[str] = []
            piece_tokens = 0
            for sentence in self._sentences(block.text):
                tokens = estimate_tokens(sentence)
                if piece and piece_tokens + tokens > budget:
                    result.append(_Block(" ".join(piece), block.start, block.section))
                    piece, piece_tokens = [], 0
                piece.append(sentence)
                piece_tokens += tokens
            if piece:
                result.append(_Block(" ".join(piece), block.start, block.section))
        return result

    def _sentences(self, text: str) -> list[str]:
        sentences: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Hard-wrap pathological sentences (tables, lists without
            # punctuation) so no single piece exceeds the budget.
            limit = int(self.max_tokens * 0.8) * 4
            while len(sentence) > limit:
                cut = sentence.rfind(" ", 0, limit)
                cut = cut if cut > 0 else limit
                sentences.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                sentences.append(sentence)
        return sentences

    def _tail(self, text: str) -> str:
        if not text or self.overlap_tokens <= 0:
            return ""
        limit = self.overlap_tokens * 4
        if len(text) <= limit:
            return text
        tail = text[-limit:]
        space = tail.find(" ")
        return tail[space + 1:] if 0 <= space < 40 else tail
