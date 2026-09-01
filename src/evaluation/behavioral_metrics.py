from __future__ import annotations

from src.validation.behavioral.trace_generator import generate_traces
from src.validation.behavioral.trace_validator import trace_similarity


def behavioral_similarity(generated, gold):
    gen_traces = generate_traces(generated)
    gold_traces = generate_traces(gold)
    return {
        "trace_similarity": trace_similarity(gen_traces, gold_traces),
        "generated_trace_count": len(gen_traces),
        "gold_trace_count": len(gold_traces),
    }
