"""LLM cost estimation.

A small static price table covering the providers we ship adapters for.
Prices change — callers should treat the output as an estimate, not an
invoice. Surfaces as `RunRecord.metrics.cost_usd` and in observability logs.
"""
from __future__ import annotations

from typing import Optional

# (provider, model_prefix) -> (USD per million input tokens, USD per million output tokens).
# Prefix match so "claude-opus-4-7" maps to ("anthropic", "claude-opus").
# Keep this list small and current; update on provider price changes.
PRICES_USD_PER_MILLION: dict[tuple[str, str], tuple[float, float]] = {
    # --- Anthropic ---
    ("anthropic", "claude-opus"):   (15.00, 75.00),
    ("anthropic", "claude-sonnet"): (3.00, 15.00),
    ("anthropic", "claude-haiku"):  (0.80, 4.00),
    # --- OpenAI ---
    ("openai", "gpt-4o-mini"):      (0.15, 0.60),
    ("openai", "gpt-4o"):           (2.50, 10.00),
    ("openai", "gpt-4-turbo"):      (10.00, 30.00),
    ("openai", "gpt-4"):            (30.00, 60.00),
    # --- Groq (public pricing; verify on https://groq.com/pricing) ---
    ("groq", "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("groq", "llama-3.1-70b"):           (0.59, 0.79),
    ("groq", "llama-3.1-8b-instant"):    (0.05, 0.08),
    ("groq", "llama-3-70b"):             (0.59, 0.79),
    ("groq", "gemma2-9b-it"):            (0.20, 0.20),
    ("groq", "gemma-7b-it"):             (0.07, 0.07),
}


def _find_price(provider: Optional[str], model: Optional[str]) -> Optional[tuple[float, float]]:
    if not provider or not model:
        return None
    p = provider.lower()
    m = model.lower()
    # Prefer the longest prefix match so 'gpt-4o-mini' beats 'gpt-4'.
    matches = [(prov, prefix) for (prov, prefix) in PRICES_USD_PER_MILLION if prov == p and prefix in m]
    if not matches:
        return None
    matches.sort(key=lambda x: -len(x[1]))
    return PRICES_USD_PER_MILLION[matches[0]]


def estimate_cost_usd(
    *,
    provider: Optional[str],
    model: Optional[str],
    tokens_in: int,
    tokens_out: int,
) -> float:
    """Estimate USD cost for an LLM call. Returns 0.0 for unknown models."""
    p = _find_price(provider, model)
    if p is None:
        return 0.0
    in_per_m, out_per_m = p
    cost = (tokens_in / 1_000_000.0) * in_per_m + (tokens_out / 1_000_000.0) * out_per_m
    return round(cost, 4)
