"""Observability tests: cost table, metered LLM, logger config."""
from __future__ import annotations

import pytest

from job_scout.llm import FakeLLM
from job_scout.observability import (
    MeteredLLM,
    configure_logging,
    estimate_cost_usd,
    get_logger,
)


class TestEstimateCost:
    @pytest.mark.parametrize(
        "provider,model,tokens_in,tokens_out,expected",
        [
            ("anthropic", "claude-opus-4-7", 0, 0, 0.0),
            ("anthropic", "claude-opus-4-7", 1_000_000, 0, 15.00),
            ("anthropic", "claude-opus-4-7", 0, 1_000_000, 75.00),
            ("anthropic", "claude-sonnet-4-6", 1_000_000, 1_000_000, 18.00),
            ("openai", "gpt-4o", 1_000_000, 0, 2.50),
            ("openai", "gpt-4o-mini", 1_000_000, 1_000_000, 0.75),
            ("groq", "llama-3.3-70b-versatile", 1_000_000, 1_000_000, 1.38),
            ("groq", "llama-3.1-8b-instant", 1_000_000, 1_000_000, 0.13),
            ("groq", "gemma2-9b-it", 1_000_000, 1_000_000, 0.40),
        ],
    )
    def test_known_provider_model(self, provider, model, tokens_in, tokens_out, expected):
        assert estimate_cost_usd(
            provider=provider, model=model, tokens_in=tokens_in, tokens_out=tokens_out
        ) == pytest.approx(expected, abs=1e-4)

    def test_unknown_model_is_zero(self):
        # Unknown model -> 0 cost (we don't guess prices).
        assert estimate_cost_usd(
            provider="anthropic", model="never-shipped", tokens_in=100, tokens_out=100
        ) == 0.0

    def test_none_provider_or_model_is_zero(self):
        assert estimate_cost_usd(provider=None, model="gpt-4o", tokens_in=1, tokens_out=1) == 0.0
        assert estimate_cost_usd(provider="openai", model=None, tokens_in=1, tokens_out=1) == 0.0

    def test_longest_prefix_wins(self):
        # 'gpt-4o-mini' must hit the mini price, not the 'gpt-4' / 'gpt-4o' entries.
        priced = estimate_cost_usd(
            provider="openai", model="gpt-4o-mini-2026", tokens_in=1_000_000, tokens_out=0
        )
        assert priced == pytest.approx(0.15, abs=1e-4)


class TestMeteredLLM:
    def test_accumulates_tokens_and_calls(self):
        inner = FakeLLM(responses=["a", "b"])
        m = MeteredLLM(inner)
        m.complete("sys", "first")
        m.complete("sys", "second")
        assert m.calls == 2
        assert m.tokens_in > 0
        assert m.tokens_out > 0

    def test_passes_through_inner_attributes(self):
        inner = FakeLLM(responses=["x"])
        m = MeteredLLM(inner)
        assert m.name == inner.name

    def test_returns_inner_response_unchanged(self):
        inner = FakeLLM(responses=["the actual text"])
        m = MeteredLLM(inner)
        out = m.complete("sys", "user")
        assert out.text == "the actual text"


class TestLogger:
    def test_get_logger_returns_a_logger(self):
        # Idempotent configuration; just verify we get a usable logger.
        configure_logging(level="INFO")
        log = get_logger("test")
        # structlog loggers expose info/warning/error/etc.
        assert callable(getattr(log, "info", None))
        assert callable(getattr(log, "warning", None))
