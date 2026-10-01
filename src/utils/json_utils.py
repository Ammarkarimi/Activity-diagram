from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        pruned = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in pruned.items() if v not in ("", None, [], {})}
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def prompt_json(value: Any) -> str:
    """Compact JSON for LLM prompts: drops empty fields, no indentation.

    Accepts pydantic models or lists of them. Keeps prompts small when a
    module carries dozens of requirements or a large diagram.
    """
    if isinstance(value, list):
        data = [v.model_dump(mode="json") if hasattr(v, "model_dump") else v for v in value]
    elif hasattr(value, "model_dump"):
        data = value.model_dump(mode="json")
    else:
        data = value
    return json.dumps(_prune(data), ensure_ascii=False, separators=(",", ":"))
