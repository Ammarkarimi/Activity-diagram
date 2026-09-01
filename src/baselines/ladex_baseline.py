from __future__ import annotations

from src.baselines.self_refine import run as self_refine_run


def run(requirement_text: str, max_rounds: int = 2, model: str | None = None):
    """Experimental LADEX-style generate -> critique -> refine scaffold.

    This is not the authors' official LADEX implementation.
    Use the published LADEX paper to reproduce its exact protocol for the paper.
    """
    return self_refine_run(requirement_text, max_rounds=max_rounds, model=model)
