"""Token usage and cost accounting for LLM calls.

Observation only: every ``OpenAIClient.complete`` call (including retries that
failed after the API answered, since those are billed too) is recorded with the
token counts the API reports. Nothing here influences the pipeline.

Each record is tagged with the agent (the structured-output class name) and with
the caller's scope, set through context variables:

- ``run``: one ``MultiAgentPipeline.run`` (a module in hierarchical mode);
- ``phase``: ``setup``, ``iteration_00``, ``iteration_01``, ..., ``finalise``.

Context variables are per thread, so modules running in parallel on a shared
client are still attributed correctly.

Cost uses ``configs/pricing.json`` (USD per 1M tokens per model):

    {"gpt-5.4-mini": {"input": 0.0, "cached_input": 0.0, "output": 0.0}}

A model missing from the file gets token counts but ``cost_usd: null``.
"""
from __future__ import annotations

import contextvars
import json
import threading
from pathlib import Path
from typing import Any

PRICING_FILE = Path(__file__).resolve().parents[2] / "configs" / "pricing.json"

_run: contextvars.ContextVar[str | None] = contextvars.ContextVar("usage_run", default=None)
_phase: contextvars.ContextVar[str | None] = contextvars.ContextVar("usage_phase", default=None)

# Structured-output class -> the agent that requests it.
AGENT_NAMES = {
    "RequirementExtractionResponse": "Requirement Agent",
    "ActorList": "Actor Agent",
    "DecompositionPlan": "Decomposition Agent",
    "ActivityPlan": "Planning Agent",
    "ActivityDiagram": "Generator Agent",
    "RefinementResult": "Refinement Agent",
    "ReviewResult": "Semantic Reviewer",
    "FeedbackResult": "Feedback Agent",
    "RepairResult": "Repair Agent",
    "LaneMapping": "Lane Mapping Agent",
}

TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")


def set_run(run_id: str | None) -> None:
    _run.set(run_id)


def set_phase(phase: str | None) -> None:
    _phase.set(phase)


def load_pricing() -> dict[str, dict[str, float]]:
    try:
        return json.loads(PRICING_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cost_usd(record: dict[str, Any], pricing: dict[str, dict[str, float]]) -> float | None:
    price = pricing.get(record["model"])
    if not price or price.get("input") is None or price.get("output") is None:
        return None
    cached = record["cached_input_tokens"]
    uncached = record["input_tokens"] - cached
    cached_rate = price.get("cached_input", price["input"])
    # Reasoning tokens are part of output_tokens and billed as output.
    return (
        uncached * price["input"]
        + cached * cached_rate
        + record["output_tokens"] * price["output"]
    ) / 1_000_000


def summarize(records: list[dict[str, Any]], pricing: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    """Totals for a list of records; ``cost_usd`` is None if any model is unpriced."""
    pricing = load_pricing() if pricing is None else pricing
    total: dict[str, Any] = {field: 0 for field in TOKEN_FIELDS}
    total["calls"] = len(records)
    total["failed_calls"] = sum(1 for r in records if r.get("failed"))
    cost: float | None = 0.0
    for record in records:
        for field in TOKEN_FIELDS:
            total[field] += record[field]
        c = cost_usd(record, pricing)
        cost = None if c is None or cost is None else cost + c
    total["total_tokens"] = total["input_tokens"] + total["output_tokens"]
    total["cost_usd"] = round(cost, 6) if cost is not None else None
    return total


def group_by(records: list[dict[str, Any]], key: str, pricing: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    pricing = load_pricing() if pricing is None else pricing
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record.get(key) or "other"), []).append(record)
    return {name: summarize(items, pricing) for name, items in groups.items()}


def report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Total, per phase and per agent."""
    pricing = load_pricing()
    return {
        "total": summarize(records, pricing),
        "by_phase": group_by(records, "phase", pricing),
        "by_agent": group_by(records, "agent", pricing),
        "pricing_source": str(PRICING_FILE) if pricing else None,
    }


class UsageTracker:
    """Thread-safe list of usage records for one client."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []

    def record(self, model: str, agent: str, usage: Any, failed: bool = False) -> None:
        if usage is None:
            return
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        entry = {
            "model": model,
            "agent": AGENT_NAMES.get(agent, agent),
            "run": _run.get(),
            "phase": _phase.get(),
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "cached_input_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
            "failed": failed,
        }
        with self._lock:
            self._records.append(entry)

    def records(self, run: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._records)
        return items if run is None else [r for r in items if r["run"] == run]

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def since(self, index: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records[index:])
