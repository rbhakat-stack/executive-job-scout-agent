"""Freshness inference for the Validation Agent.

Decision tree (most authoritative first):

  1. ATS / JSON-LD `datePosted`  -> evidence.source = 'json_ld_datePosted'
  2. <meta property="article:published_time">  -> 'meta_published_time'
  3. Search-engine result date (if the search provider returned one)
     -> 'search_engine_date' (confidence dropped to 0.6)
  4. None of the above           -> FreshnessLabel.UNKNOWN, no evidence

The labels RECENT and OLDER are computed by comparing `posted_at` against
`today - max_age_days`. The Freshness schema's invariant (RECENT/OLDER
require evidence) is what guarantees we can't surface a fake "recent" date.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from job_scout.parsers.ats import ExtractedJob
from job_scout.schemas import Freshness, FreshnessEvidence, FreshnessLabel


def infer_freshness(
    extracted: ExtractedJob,
    *,
    search_engine_date_iso: Optional[str] = None,
    search_engine_date: Optional[date] = None,
    today: date,
    max_age_days: int,
) -> Freshness:
    """Return a `Freshness` for the extracted job.

    `search_engine_date_iso` and `search_engine_date` are alternatives;
    either may be supplied. They represent the date the search provider
    associated with the result — used only when the page itself has no date.
    """
    # Prefer the date extracted from the page itself.
    posted_at = extracted.posted_at
    evidence_source = extracted.posted_at_source
    evidence_snippet = extracted.posted_at_snippet
    confidence = 1.0

    # Fall back to the search-engine-reported date if the page has no date.
    if posted_at is None and (search_engine_date or search_engine_date_iso):
        if search_engine_date is None and search_engine_date_iso:
            # Best-effort parse; if it fails we drop to UNKNOWN.
            try:
                posted_at = date.fromisoformat(
                    search_engine_date_iso.replace("Z", "+00:00")[:10]
                )
            except ValueError:
                posted_at = None
        else:
            posted_at = search_engine_date
        if posted_at is not None:
            evidence_source = "search_engine_date"
            evidence_snippet = (
                search_engine_date_iso
                if search_engine_date_iso
                else posted_at.isoformat()
            )
            confidence = 0.6

    if posted_at is None:
        return Freshness(label=FreshnessLabel.UNKNOWN)

    cutoff = today - timedelta(days=max_age_days)
    label = FreshnessLabel.RECENT if posted_at >= cutoff else FreshnessLabel.OLDER

    return Freshness(
        label=label,
        posted_at=posted_at,
        evidence=FreshnessEvidence(
            source=evidence_source or "unknown",
            snippet=evidence_snippet,
            confidence=confidence,
        ),
    )
