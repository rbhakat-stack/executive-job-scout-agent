"""Schema contract tests.

These pin down the invariants the Validation + Red Team agents rely on:
1. `Freshness` cannot be RECENT or OLDER without evidence (the 'no stale
   jobs as active' invariant).
2. `SearchCriteria` rejects nonsensical comp ranges.
3. `ValidatedJob` requires the structural fields that the Scoring Agent needs.
4. All schemas are JSON-round-trippable (used by the DB layer).
"""
from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.schemas import (
    ATS,
    CandidateProfile,
    Citation,
    EvidenceBundle,
    Freshness,
    FreshnessEvidence,
    FreshnessLabel,
    JobSignals,
    JobStatus,
    RawJobLead,
    SearchCriteria,
    SearchProviderName,
    SearchQuery,
    SearchStrategy,
    SeniorityLevel,
    ValidatedJob,
    ValidationResult,
    WorkMode,
)


# ---------------------------------------------------------------------------
# Freshness invariant
# ---------------------------------------------------------------------------

class TestFreshnessInvariant:
    def test_unknown_without_evidence_is_ok(self):
        f = Freshness(label=FreshnessLabel.UNKNOWN)
        assert f.label is FreshnessLabel.UNKNOWN
        assert f.evidence is None

    def test_recent_without_evidence_is_rejected(self):
        with pytest.raises(ValueError, match="evidence"):
            Freshness(label=FreshnessLabel.RECENT, posted_at=date(2026, 5, 20))

    def test_older_without_evidence_is_rejected(self):
        with pytest.raises(ValueError, match="evidence"):
            Freshness(label=FreshnessLabel.OLDER, posted_at=date(2024, 1, 1))

    def test_recent_with_evidence_is_ok(self):
        f = Freshness(
            label=FreshnessLabel.RECENT,
            posted_at=date(2026, 5, 20),
            evidence=FreshnessEvidence(source="json_ld_datePosted", snippet="2026-05-20"),
        )
        assert f.label is FreshnessLabel.RECENT


# ---------------------------------------------------------------------------
# SearchCriteria validation
# ---------------------------------------------------------------------------

class TestSearchCriteria:
    def test_defaults(self):
        c = SearchCriteria()
        assert c.max_age_days == 14
        assert c.allow_older is False
        assert c.prioritize_urgent is True
        assert c.min_match_score == 60
        # Default to all work modes allowed
        assert set(c.work_modes) == {WorkMode.REMOTE, WorkMode.HYBRID, WorkMode.ONSITE}

    def test_comp_max_must_be_geq_comp_min(self):
        with pytest.raises(ValidationError):
            SearchCriteria(comp_min_usd=300_000, comp_max_usd=200_000)

    def test_max_age_days_bounds(self):
        with pytest.raises(ValidationError):
            SearchCriteria(max_age_days=0)
        with pytest.raises(ValidationError):
            SearchCriteria(max_age_days=10_000)


# ---------------------------------------------------------------------------
# CandidateProfile round-trip
# ---------------------------------------------------------------------------

class TestCandidateProfile:
    def test_minimum_required_fields(self):
        p = CandidateProfile(
            resume_text_sha256="a" * 64,
            summary="Senior life-sciences tech leader with 20+ years.",
            seniority_level=SeniorityLevel.SVP,
        )
        # Round trip through JSON
        s = p.model_dump_json()
        p2 = CandidateProfile.model_validate_json(s)
        assert p2.resume_text_sha256 == p.resume_text_sha256
        assert p2.seniority_level is SeniorityLevel.SVP

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            CandidateProfile(
                resume_text_sha256="a" * 64,
                summary="x" * 20,
                seniority_level=SeniorityLevel.VP,
                rogue_field="should not be allowed",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# ValidatedJob structural requirements
# ---------------------------------------------------------------------------

def _raw_lead() -> RawJobLead:
    return RawJobLead(
        title_guess="VP AI Transformation",
        url="https://boards.greenhouse.io/acme/jobs/123",
        snippet="VP AI - life sciences",
        source_provider=SearchProviderName.TAVILY,
        source_query=SearchQuery(
            strategy=SearchStrategy.EXACT_TITLE,
            text='"VP AI" "life sciences"',
        ),
        ats_guess=ATS.GREENHOUSE,
    )


def _validation_ok() -> ValidationResult:
    return ValidationResult(
        live=True,
        http_status=200,
        final_url="https://boards.greenhouse.io/acme/jobs/123",
    )


def _freshness_recent() -> Freshness:
    return Freshness(
        label=FreshnessLabel.RECENT,
        posted_at=date(2026, 5, 20),
        evidence=FreshnessEvidence(source="ats_field", snippet="2026-05-20T00:00:00Z"),
    )


class TestValidatedJob:
    def test_happy_path(self):
        job = ValidatedJob(
            raw_lead=_raw_lead(),
            validation=_validation_ok(),
            canonical_url="https://boards.greenhouse.io/acme/jobs/123",
            apply_url="https://boards.greenhouse.io/acme/jobs/123/apply",
            dedup_hash="d" * 64,
            title="VP AI Transformation",
            company="Acme Bio",
            body_text="We are hiring a VP of AI Transformation to lead our..." + ("x" * 20),
            freshness=_freshness_recent(),
        )
        assert job.work_mode is WorkMode.UNKNOWN
        assert job.status is JobStatus.ACTIVE

    def test_body_text_must_be_substantive(self):
        with pytest.raises(ValidationError):
            ValidatedJob(
                raw_lead=_raw_lead(),
                validation=_validation_ok(),
                canonical_url="https://boards.greenhouse.io/acme/jobs/123",
                apply_url="https://boards.greenhouse.io/acme/jobs/123/apply",
                dedup_hash="d" * 64,
                title="VP AI",
                company="Acme",
                body_text="too short",   # < 20 chars
                freshness=_freshness_recent(),
            )

    def test_dedup_hash_too_short_is_rejected(self):
        with pytest.raises(ValidationError):
            ValidatedJob(
                raw_lead=_raw_lead(),
                validation=_validation_ok(),
                canonical_url="https://boards.greenhouse.io/acme/jobs/123",
                apply_url="https://boards.greenhouse.io/acme/jobs/123/apply",
                dedup_hash="short",
                title="VP AI",
                company="Acme",
                body_text="A reasonably long job description follows here.",
                freshness=_freshness_recent(),
            )


# ---------------------------------------------------------------------------
# EvidenceBundle
# ---------------------------------------------------------------------------

class TestEvidenceBundle:
    def test_keys_collects_claim_keys(self):
        eb = EvidenceBundle(
            citations=[
                Citation(
                    claim_key="match.industry",
                    quote="life sciences",
                    source_url="https://example.com/jobs/1",
                ),
                Citation(
                    claim_key="urgency.posted_recently",
                    quote="Posted 3 days ago",
                    source_url="https://example.com/jobs/1",
                ),
            ]
        )
        assert eb.keys() == {"match.industry", "urgency.posted_recently"}

    def test_citation_quote_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            Citation(
                claim_key="x",
                quote="",
                source_url="https://example.com/jobs/1",
            )
