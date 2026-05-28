"""Neutral home for the LLM error type.

Lives outside any concrete adapter so tracebacks read
`src.llm.errors.LLMError` regardless of which provider raised it.
"""
from __future__ import annotations


class LLMError(Exception):
    """Raised when an LLM call fails (auth, transport, rate limit, parse).

    The Scoring Agent catches this and falls back to the deterministic
    rationale rather than aborting the run.
    """
