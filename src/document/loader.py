from __future__ import annotations

from pathlib import Path


def load_document(path: str | Path) -> str:
    """Load a requirement specification as plain text.

    Supports .txt/.md (and any other text file), .pdf (requires ``pypdf``)
    and .docx (requires ``python-docx``).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Reading PDF files requires `pip install pypdf`.") from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if suffix == ".docx":
        try:
            import docx
        except ImportError as exc:
            raise RuntimeError("Reading DOCX files requires `pip install python-docx`.") from exc
        document = docx.Document(str(path))
        lines = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                lines.append("")
                continue
            style = (paragraph.style.name or "").lower() if paragraph.style else ""
            if style.startswith("heading"):
                level = "".join(ch for ch in style if ch.isdigit()) or "1"
                lines.append("")
                lines.append("#" * int(level) + " " + text)
                lines.append("")
            else:
                lines.append(text)
        return "\n".join(lines)

    return path.read_text(encoding="utf-8", errors="replace")
