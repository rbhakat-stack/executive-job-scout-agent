"""LLM factory tests.

These pin the provider routing (anthropic / openai / groq / none) without
actually invoking any LLM. No real keys, no network.
"""
from __future__ import annotations

import pytest

from src.config import SessionKeyStore, Settings
from src.llm import LLMError, build_llm


class TestBuildLLM:
    def test_returns_none_when_no_key_for_anthropic(self):
        # Empty session + Settings with no key -> None (Scoring Agent
        # falls back to deterministic rationale).
        s = Settings(ANTHROPIC_API_KEY=None)
        assert build_llm(
            provider="anthropic", session=SessionKeyStore(), settings=s
        ) is None

    def test_returns_none_when_no_key_for_openai(self):
        s = Settings(OPENAI_API_KEY=None)
        assert build_llm(
            provider="openai", session=SessionKeyStore(), settings=s
        ) is None

    def test_returns_none_when_no_key_for_groq(self):
        s = Settings(GROQ_API_KEY=None)
        assert build_llm(
            provider="groq", session=SessionKeyStore(), settings=s
        ) is None

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            build_llm(provider="cohere", session=SessionKeyStore())

    def test_case_insensitive_provider(self):
        # 'GROQ' should resolve like 'groq'.
        s = Settings(GROQ_API_KEY=None)
        assert build_llm(
            provider="GROQ", session=SessionKeyStore(), settings=s
        ) is None


class TestLLMErrorLocation:
    """Pin LLMError's home so tracebacks read sensibly.

    Regression test for the misleading 'src.llm.anthropic_client.LLMError'
    in OpenAI tracebacks - the class now lives in `src.llm.errors`.
    """

    def test_lives_in_neutral_module(self):
        assert LLMError.__module__ == "src.llm.errors"

    def test_all_adapters_use_the_same_exception(self):
        # Importing from each adapter module should yield the same class.
        from src.llm.anthropic_client import LLMError as A
        from src.llm.groq_client import LLMError as G
        from src.llm.openai_client import LLMError as O
        assert A is LLMError
        assert G is LLMError
        assert O is LLMError
