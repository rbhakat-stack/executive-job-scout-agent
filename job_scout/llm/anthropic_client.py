"""Anthropic LLM adapter.

Lazy-imports `anthropic` so the package only needs to be installed when this
adapter is actually constructed at runtime. Tests use `FakeLLM`.
"""
from __future__ import annotations

from typing import Optional

from .base import LLMResponse
from .errors import LLMError


class AnthropicLLM:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-opus-4-7",
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise LLMError("Anthropic API key is required.")
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMError(
                "anthropic package is not installed. "
                "Install with `pip install anthropic`."
            ) from e
        self._client = Anthropic(api_key=api_key)
        self.model: str = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> LLMResponse:
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:  # SDK raises various exception types
            raise LLMError(f"Anthropic API error: {e}") from e

        # `messages.create` returns content as a list of content blocks.
        text_parts: list[str] = []
        for block in msg.content or []:
            t = getattr(block, "text", None)
            if t:
                text_parts.append(t)
        text = "".join(text_parts)

        usage = getattr(msg, "usage", None)
        return LLMResponse(
            text=text,
            tokens_in=getattr(usage, "input_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "output_tokens", 0) if usage else 0,
            model=self.model,
        )
