"""Deterministic in-process search provider for tests + offline runs."""
from __future__ import annotations

from typing import Callable, Optional

from src.schemas import SearchProviderName

from .base import SearchProvider, SearchProviderResult


class FakeSearchProvider:
    """In-process search provider for tests.

    Provide either a `results_by_query` map (exact-string match) or a
    `responder` callable. Exactly one is required.
    """

    name = SearchProviderName.FAKE

    def __init__(
        self,
        *,
        results_by_query: Optional[dict[str, list[SearchProviderResult]]] = None,
        responder: Optional[Callable[[str], list[SearchProviderResult]]] = None,
    ) -> None:
        if (results_by_query is None) == (responder is None):
            raise ValueError(
                "FakeSearchProvider: provide exactly one of "
                "`results_by_query` or `responder`."
            )
        self._results = results_by_query
        self._responder = responder
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int = 10) -> list[SearchProviderResult]:
        self.calls.append(query)
        if self._results is not None:
            results = list(self._results.get(query, []))
        else:
            assert self._responder is not None
            results = list(self._responder(query))
        return results[:max_results]
