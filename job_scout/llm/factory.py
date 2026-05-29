"""LLM provider factory.

Returns a concrete `LLM` for the given provider name + key. Used by the UI
to construct the right adapter from the user's session-only settings.
"""
from __future__ import annotations

from typing import Optional

from job_scout.config import Settings, get_settings, resolve_secret
from job_scout.config import SessionKeyStore

from .anthropic_client import AnthropicLLM
from .base import LLM
from .errors import LLMError
from .groq_client import GroqLLM
from .openai_client import OpenAILLM


def build_llm(
    *,
    provider: str,
    session: Optional[SessionKeyStore] = None,
    settings: Optional[Settings] = None,
    model: Optional[str] = None,
) -> Optional[LLM]:
    """Construct an LLM adapter for the requested provider.

    Returns `None` if no key is available (the Scoring Agent falls back to
    deterministic rationales). Raises `LLMError` if the provider name is
    unknown or the underlying SDK import fails.
    """
    p = (provider or "").lower().strip()
    s = settings or get_settings()
    model = model or s.LLM_MODEL

    if p == "anthropic":
        key = resolve_secret("ANTHROPIC_API_KEY", session=session, settings=s)
        if not key:
            return None
        return AnthropicLLM(api_key=key, model=model)

    if p == "openai":
        key = resolve_secret("OPENAI_API_KEY", session=session, settings=s)
        if not key:
            return None
        return OpenAILLM(api_key=key, model=model)

    if p == "groq":
        key = resolve_secret("GROQ_API_KEY", session=session, settings=s)
        if not key:
            return None
        return GroqLLM(api_key=key, model=model)

    raise LLMError(f"Unknown LLM provider: {provider!r}")
