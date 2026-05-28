"""Search provider abstraction.

Agents talk to search providers through the `SearchProvider` protocol —
never directly to a vendor SDK. The `SearchProviderResult` is the normalized
shape across providers; concrete adapters translate vendor responses into it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol

from pydantic import BaseModel, ConfigDict, HttpUrl

from src.schemas import SearchProviderName


class SearchProviderResult(BaseModel):
    """A single raw search hit, normalized across providers."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: HttpUrl
    snippet: Optional[str] = None
    published_date: Optional[datetime] = None


class SearchProvider(Protocol):
    """Capability surface used by the Search Agent."""

    name: SearchProviderName

    def search(self, query: str, *, max_results: int = 10) -> list[SearchProviderResult]: ...


class SearchProviderError(Exception):
    """Raised by a provider for any failure (auth, transport, parse, rate limit)."""
