"""Groq LLM adapter (lazy import).

Uses the `groq` Python SDK, whose chat-completions surface mirrors the
OpenAI client. Lazy-imports the SDK so installs that don't need Groq
don't pay for the dependency at import time.
"""
from __future__ import annotations

from .base import LLMResponse
from .errors import LLMError


class GroqLLM:
    name = "groq"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 2048,
    ) -> None:
        if not api_key:
            raise LLMError("Groq API key is required.")
        try:
            from groq import Groq
        except ImportError as e:
            raise LLMError(
                "groq package is not installed. Install with `pip install groq`."
            ) from e
        self._client = Groq(api_key=api_key)
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
            # Classify common Groq failure modes so the user-visible error
            # in the UI points at the actual cause instead of a generic
            # "Groq API error".
            msg = str(e).lower()
            if ("rate" in msg and "limit" in msg) or "429" in msg:
                raise LLMError(
                    f"Groq rate limit hit (free tier is 30 requests/min on "
                    f"{self.model}). Wait ~60s and retry, or switch to a "
                    f"paid Groq tier. Underlying: {e}"
                ) from e
            if "401" in msg or "unauthorized" in msg or "invalid_api_key" in msg:
                raise LLMError(
                    f"Groq rejected the API key (invalid or revoked). "
                    f"Re-enter it in the Settings sidebar. Underlying: {e}"
                ) from e
            if "404" in msg or "model_not_found" in msg or "does not exist" in msg:
                raise LLMError(
                    f"Groq model '{self.model}' not found. The model may "
                    f"have been retired. Try 'llama-3.3-70b-versatile'. "
                    f"Underlying: {e}"
                ) from e
            if "quota" in msg or "402" in msg or "insufficient" in msg:
                raise LLMError(
                    f"Groq quota exhausted. Check your usage at "
                    f"console.groq.com. Underlying: {e}"
                ) from e
            raise LLMError(f"Groq API error: {e}") from e

        choice = (resp.choices or [None])[0]
        text = (choice.message.content if choice and choice.message else "") or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
            model=self.model,
        )
