"""Freshness inference tests.

These pin the 'no stale jobs as active' invariant at the *inference* level:
- Page date present and recent  -> RECENT (with evidence)
- Page date present and old     -> OLDER (with evidence)
- Page date absent but search-engine date supplied -> labeled with search_engine_date evidence
- No date anywhere              -> UNKNOWN (never RECENT)

The Freshness schema's `model_post_init` also enforces this — these tests
check the inference layer makes the right decision before the schema check.
"""
from __future__ import annotations

from datetime import date

from src.parsers.ats import ExtractedJob
from src.schemas import FreshnessLabel
from src.validation.freshness import infer_freshness


TODAY = date(2026, 5, 27)


class TestInferFreshness:
    def test_recent_with_page_date(self):
        ext = ExtractedJob(
            posted_at=date(2026, 5, 20),
            posted_at_source="json_ld_datePosted",
            posted_at_snippet="2026-05-20",
        )
        f = infer_freshness(ext, today=TODAY, max_age_days=14)
        assert f.label is FreshnessLabel.RECENT
        assert f.posted_at == date(2026, 5, 20)
        assert f.evidence is not None
        assert f.evidence.source == "json_ld_datePosted"

    def test_older_with_page_date(self):
        ext = ExtractedJob(
            posted_at=date(2025, 1, 1),
            posted_at_source="json_ld_datePosted",
            posted_at_snippet="2025-01-01",
        )
        f = infer_freshness(ext, today=TODAY, max_age_days=14)
        assert f.label is FreshnessLabel.OLDER
        assert f.evidence is not None
        assert f.evidence.confidence == 1.0

    def test_unknown_when_no_date(self):
        ext = ExtractedJob()
        f = infer_freshness(ext, today=TODAY, max_age_days=14)
        assert f.label is FreshnessLabel.UNKNOWN
        assert f.posted_at is None
        assert f.evidence is None

    def test_search_engine_date_used_when_page_has_none(self):
        ext = ExtractedJob()  # no page date
        f = infer_freshness(
            ext,
            search_engine_date=date(2026, 5, 18),
            today=TODAY,
            max_age_days=14,
        )
        assert f.label is FreshnessLabel.RECENT
        assert f.evidence is not None
        assert f.evidence.source == "search_engine_date"
        # Confidence is lower than for page-extracted dates.
        assert f.evidence.confidence == 0.6

    def test_page_date_wins_over_search_engine_date(self):
        ext = ExtractedJob(
            posted_at=date(2026, 5, 20),
            posted_at_source="json_ld_datePosted",
            posted_at_snippet="2026-05-20",
        )
        f = infer_freshness(
            ext,
            search_engine_date=date(2026, 5, 10),
            today=TODAY,
            max_age_days=14,
        )
        assert f.evidence.source == "json_ld_datePosted"
        assert f.posted_at == date(2026, 5, 20)

    def test_search_engine_date_iso_string_parsed(self):
        ext = ExtractedJob()
        f = infer_freshness(
            ext,
            search_engine_date_iso="2026-05-18T00:00:00Z",
            today=TODAY,
            max_age_days=14,
        )
        assert f.label is FreshnessLabel.RECENT
        assert f.evidence.snippet == "2026-05-18T00:00:00Z"

    def test_unparseable_search_engine_date_falls_through_to_unknown(self):
        ext = ExtractedJob()
        f = infer_freshness(
            ext,
            search_engine_date_iso="not a date",
            today=TODAY,
            max_age_days=14,
        )
        assert f.label is FreshnessLabel.UNKNOWN

    def test_boundary_inclusive(self):
        # Exactly max_age_days old should still be RECENT.
        ext = ExtractedJob(
            posted_at=date(2026, 5, 13),  # 14 days before TODAY
            posted_at_source="json_ld_datePosted",
            posted_at_snippet="2026-05-13",
        )
        f = infer_freshness(ext, today=TODAY, max_age_days=14)
        assert f.label is FreshnessLabel.RECENT
