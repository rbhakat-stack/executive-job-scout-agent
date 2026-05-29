"""Tavily search adapter.

Uses Tavily's REST API directly via httpx so:
  1. Tests can substitute an `httpx.MockTransport` (no vendor SDK to mock).
  2. We're not bound to the lifecycle of the `tavily-python` package.

Constructor accepts an optional `client` so tests inject a mock transport.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx

from job_scout.schemas import SearchProviderName

from .base import SearchProvider, SearchProviderError, SearchProviderResult

TAVILY_URL = "https://api.tavily.com/search"


class TavilySearchProvider:
    """Tavily adapter implementing the SearchProvider protocol."""

    name = SearchProviderName.TAVILY

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: int = 15,
        client: Optional[httpx.Client] = None,
    ) -> None:
        if not api_key:
            raise SearchProviderError("Tavily API key is required.")
        self._api_key = api_key
        # If the caller didn't pass a client, build one with the configured timeout.
        # Tests pass `httpx.Client(transport=httpx.MockTransport(...))`.
        self._client = client or httpx.Client(timeout=timeout_seconds)

    def search(self, query: str, *, max_results: int = 10) -> list[SearchProviderResult]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max(1, min(max_results, 20)),
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            resp = self._client.post(TAVILY_URL, json=payload)
        except httpx.HTTPError as e:
            raise SearchProviderError(f"Tavily transport error: {e}") from e

        if resp.status_code != 200:
            raise SearchProviderError(
                f"Tavily HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise SearchProviderError(f"Tavily returned non-JSON: {e}") from e

        out: list[SearchProviderResult] = []
        for item in data.get("results", []):
            url = item.get("url")
            if not url:
                continue
            try:
                out.append(
                    SearchProviderResult(
                        title=item.get("title") or "(untitled)",
                        url=url,
                        snippet=item.get("content"),
                        published_date=_parse_iso8601(item.get("published_date")),
                    )
                )
            except Exception:
                # Skip malformed entries quietly — the Validation Agent
                # would reject them anyway and one bad row should not
                # poison the whole query.
                continue
        return out


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
