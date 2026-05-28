"""OpenAI LLM adapter (lazy import)."""
from __future__ import annotations

from .base import LLMResponse
from .errors import LLMError


class OpenAILLM:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o",
        max_tokens: int = 2048,
    ) -> None:
        if not api_key:
            raise LLMError("OpenAI API key is required.")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError(
                "openai package is not installed. Install with `pip install openai`."
            ) from e
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> LLMResponse:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            raise LLMError(f"OpenAI API error: {e}") from e

        choice = (resp.choices or [None])[0]
        text = (choice.message.content if choice and choice.message else "") or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
            model=self.model,
        )
