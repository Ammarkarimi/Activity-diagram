from __future__ import annotations

from math import sqrt


def cohens_d(a, b):
    """Simple independent-sample Cohen's d without scipy."""
    if not a or not b:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / max(1, len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / max(1, len(b) - 1)
    pooled = sqrt(((len(a)-1)*va + (len(b)-1)*vb) / max(1, len(a)+len(b)-2))
    return (ma - mb) / pooled if pooled else 0.0
