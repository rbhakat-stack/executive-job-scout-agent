"""LLM provider abstraction.

Agents talk to LLMs through the `LLM` protocol — never through a concrete
provider SDK. Concrete provider adapters live in this module (added as the
milestones that consume them require). For M2 we ship the protocol + a
`FakeLLM` for tests; real provider adapters come in M5/M8.
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol

from pydantic import BaseModel, ConfigDict


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: Optional[str] = None


class LLM(Protocol):
    """Minimal capability surface used by the agents.

    `complete()` returns text. Structured outputs are parsed by the caller
    (the agent) so that schema validation is the agent's responsibility,
    not the provider's.
    """

    name: str

    def complete(self, system: str, user: str) -> LLMResponse: ...


class FakeLLM:
    """Deterministic LLM stand-in for tests.

    Provide either a `responses` queue (returned in order) or a `responder`
    callable `(system, user) -> str`. Exactly one is required.
    """

    name = "fake"

    def __init__(
        self,
        *,
        responses: Optional[list[str]] = None,
        responder: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        if (responses is None) == (responder is None):
            raise ValueError("FakeLLM: provide exactly one of `responses` or `responder`.")
        self._queue: Optional[list[str]] = list(responses) if responses else None
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        if self._queue is not None:
            if not self._queue:
                raise RuntimeError("FakeLLM: response queue exhausted.")
            text = self._queue.pop(0)
        else:
            assert self._responder is not None
            text = self._responder(system, user)
        # Token estimates are deliberately rough; tests that pin specific
        # token counts should use a custom responder.
        return LLMResponse(
            text=text,
            tokens_in=max(1, len(user) // 4),
            tokens_out=max(1, len(text) // 4),
            model="fake",
        )
