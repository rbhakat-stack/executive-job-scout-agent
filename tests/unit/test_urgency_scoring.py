"""Urgency feature tests."""
from __future__ import annotations

from datetime import date

import pytest

from job_scout.schemas import (
    ATS,
    Freshness,
    FreshnessEvidence,
    FreshnessLabel,
    JobSignals,
    JobStatus,
    RawJobLead,
    SearchProviderName,
    SearchQuery,
    SearchStrategy,
    ValidatedJob,
    ValidationResult,
    WorkMode,
)
from job_scout.scoring.urgency import compute_urgency_features


TODAY = date(2026, 5, 27)


def _job(
    *,
    posted_at: date | None = date(2026, 5, 25),
    signals: JobSignals | None = None,
    ats: ATS | None = ATS.GREENHOUSE,
) -> ValidatedJob:
    url = "https://boards.greenhouse.io/acme/jobs/1"
    return ValidatedJob(
        raw_lead=RawJobLead(
            title_guess="x",
            url=url,
            source_provider=SearchProviderName.TAVILY,
            source_query=SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text="x"),
        ),
        validation=ValidationResult(live=True, http_status=200, final_url=url),
        canonical_url=url,
        apply_url=url,
        dedup_hash="d" * 64,
        title="VP AI",
        company="Acme",
        body_text="A substantive job description that easily clears the minimum length.",
        ats=ats,
        work_mode=WorkMode.REMOTE,
        freshness=Freshness(
            label=FreshnessLabel.RECENT if posted_at else FreshnessLabel.UNKNOWN,
            posted_at=posted_at,
            evidence=(
                FreshnessEvidence(source="json_ld_datePosted", snippet=str(posted_at))
                if posted_at
                else None
            ),
        ),
        signals=signals or JobSignals(),
        status=JobStatus.ACTIVE,
    )


def _f(features, name):
    for f in features:
        if f.name == name:
            return f
    raise AssertionError(f"missing feature {name}")


class TestTotalWeight:
    def test_weights_sum_to_100(self):
        features = compute_urgency_features(_job(), today=TODAY)
        assert sum(f.weight for f in features) == 100


class TestRecency:
    def test_within_7_days_max_credit(self):
        features = compute_urgency_features(_job(posted_at=date(2026, 5, 22)), today=TODAY)
        assert _f(features, "posted_within_7d").contribution == 25.0
        assert _f(features, "posted_within_14d").contribution == 15.0

    def test_exactly_7_days_boundary_inclusive(self):
        features = compute_urgency_features(_job(posted_at=date(2026, 5, 20)), today=TODAY)
        assert _f(features, "posted_within_7d").contribution == 25.0

    def test_within_8_to_14_days(self):
        features = compute_urgency_features(_job(posted_at=date(2026, 5, 15)), today=TODAY)
        assert _f(features, "posted_within_7d").contribution == 0.0
        assert _f(features, "posted_within_14d").contribution == 15.0

    def test_older_than_14_days_zero(self):
        features = compute_urgency_features(_job(posted_at=date(2026, 1, 1)), today=TODAY)
        assert _f(features, "posted_within_7d").contribution == 0.0
        assert _f(features, "posted_within_14d").contribution == 0.0

    def test_unknown_posted_at_zero(self):
        features = compute_urgency_features(_job(posted_at=None), today=TODAY)
        assert _f(features, "posted_within_7d").contribution == 0.0
        assert _f(features, "posted_within_14d").contribution == 0.0


class TestSignals:
    def test_urgency_phrase_full(self):
        job = _job(signals=JobSignals(urgency_phrases=["urgently hiring"]))
        f = _f(compute_urgency_features(job, today=TODAY), "urgency_phrase_present")
        assert f.contribution == 15.0
        assert "urgency.phrase" in f.evidence_keys

    def test_recruiter_contact(self):
        job = _job(signals=JobSignals(recruiter_contact="jane@acme.com"))
        f = _f(compute_urgency_features(job, today=TODAY), "recruiter_contact_listed")
        assert f.contribution == 10.0

    def test_multiple_openings(self):
        job = _job(signals=JobSignals(multiple_openings=True))
        f = _f(compute_urgency_features(job, today=TODAY), "multiple_openings")
        assert f.contribution == 10.0

    def test_transformation_phrase(self):
        job = _job(signals=JobSignals(transformation_phrases=["ai transformation"]))
        f = _f(compute_urgency_features(job, today=TODAY), "transformation_phrase_present")
        assert f.contribution == 10.0


class TestAtsReliability:
    @pytest.mark.parametrize(
        "ats",
        [ATS.GREENHOUSE, ATS.LEVER, ATS.WORKDAY, ATS.ASHBY, ATS.SMARTRECRUITERS, ATS.ICIMS],
    )
    def test_known_ats_full_credit(self, ats):
        f = _f(compute_urgency_features(_job(ats=ats), today=TODAY), "ats_source_reliable")
        assert f.contribution == 15.0
        assert "urgency.source_ats" in f.evidence_keys

    def test_unknown_ats_partial(self):
        f = _f(compute_urgency_features(_job(ats=None), today=TODAY), "ats_source_reliable")
        # Partial credit, not zero (a non-ATS career page isn't worthless).
        assert f.contribution == 5.0
        assert f.evidence_keys == []
