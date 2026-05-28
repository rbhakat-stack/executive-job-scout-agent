"""Search Agent.

Executes a `SearchPlan` against a `SearchProvider` and returns a flat list
of `RawJobLead`. Performs URL-level de-duplication and ATS detection from
URL host. Provider failures on a single query are recorded and skipped —
they do not abort the run, because partial results beat no results.

Final de-duplication (by content hash) is the Validation Agent's job.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from src.schemas import ATS, RawJobLead, SearchPlan
from src.search_providers.base import SearchProvider, SearchProviderError

# URL host substring -> ATS detection. Order matters: more specific hosts
# (boards.greenhouse.io) before more general ones (greenhouse.io).
_ATS_HOST_RULES: tuple[tuple[str, ATS], ...] = (
    ("boards.greenhouse.io", ATS.GREENHOUSE),
    ("greenhouse.io",        ATS.GREENHOUSE),
    ("jobs.lever.co",        ATS.LEVER),
    ("lever.co",             ATS.LEVER),
    ("myworkdayjobs.com",    ATS.WORKDAY),
    ("workday.com",          ATS.WORKDAY),
    ("jobs.ashbyhq.com",     ATS.ASHBY),
    ("ashbyhq.com",          ATS.ASHBY),
    ("jobs.smartrecruiters.com", ATS.SMARTRECRUITERS),
    ("smartrecruiters.com",  ATS.SMARTRECRUITERS),
    ("icims.com",            ATS.ICIMS),
)


def detect_ats(url: str) -> Optional[ATS]:
    """Return the detected ATS for a URL, or None if no rule matches."""
    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    for pattern, ats in _ATS_HOST_RULES:
        if pattern in host:
            return ats
    return None


class SearchAgentError(Exception):
    pass


class SearchAgent:
    """Executes a search plan against a provider and emits RawJobLeads."""

    def __init__(
        self,
        provider: SearchProvider,
        *,
        max_results_per_query: int = 10,
    ) -> None:
        self._provider = provider
        self._max_per_q = max_results_per_query
        # Recorded errors: list of (query_text, error_message). Surfaced
        # through the orchestrator's rejection log.
        self.errors: list[tuple[str, str]] = []

    def run(self, plan: SearchPlan) -> list[RawJobLead]:
        seen_urls: set[str] = set()
        leads: list[RawJobLead] = []

        for query in plan.queries:
            try:
                results = self._provider.search(
                    query.text, max_results=self._max_per_q
                )
            except SearchProviderError as e:
                self.errors.append((query.text, str(e)))
                continue

            for r in results:
                url_s = str(r.url)
                if url_s in seen_urls:
                    continue
                seen_urls.add(url_s)
                leads.append(
                    RawJobLead(
                        title_guess=(r.title or "(unknown)").strip()[:200],
                        url=r.url,
                        snippet=r.snippet,
                        source_provider=self._provider.name,
                        source_query=query,
                        ats_guess=detect_ats(url_s),
                        search_engine_date=r.published_date,
                    )
                )

        return leads
