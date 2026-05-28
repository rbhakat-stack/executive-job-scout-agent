"""MeteredLLM: a transparent decorator that counts tokens + calls.

The Scoring Agent (and any other LLM-using agent) treats this as an `LLM`
exactly like a concrete provider. The orchestrator reads the accumulated
counters at the end of a run to populate `RunMetrics`.
"""
from __future__ import annotations

from src.llm.base import LLM, LLMResponse


class MeteredLLM:
    """LLM wrapper that counts tokens, output text, and call count."""

    def __init__(self, inner: LLM) -> None:
        self._inner = inner
        # Pass-through attributes used by agents/observability.
        self.name: str = getattr(inner, "name", "wrapped")
        self.model: str | None = getattr(inner, "model", None)
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.calls: int = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        resp = self._inner.complete(system, user)
        self.tokens_in += int(resp.tokens_in or 0)
        self.tokens_out += int(resp.tokens_out or 0)
        self.calls += 1
        return resp
