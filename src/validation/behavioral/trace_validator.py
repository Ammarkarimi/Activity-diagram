from __future__ import annotations

from difflib import SequenceMatcher


def trace_similarity(generated_traces, gold_traces):
    if not generated_traces or not gold_traces:
        return 0.0
    scores = []
    for gen in generated_traces:
        best = 0.0
        for gold in gold_traces:
            a = " | ".join(gen).lower()
            b = " | ".join(gold).lower()
            best = max(best, SequenceMatcher(None, a, b).ratio())
        scores.append(best)
    return sum(scores) / len(scores)
