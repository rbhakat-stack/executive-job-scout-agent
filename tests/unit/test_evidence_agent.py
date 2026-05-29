"""Evidence Agent tests.

These pin the citation contract: every non-zero feature's `evidence_keys`
must have a corresponding `Citation` in the bundle. Tests are organized
per claim_key so coverage gaps are easy to diagnose.
"""
from __future__ import annotations

from datetime import date

import pytest

from job_scout.agents.evidence import EvidenceAgent, coverage_gaps
from job_scout.agents.scoring import ScoringAgent
from job_scout.schemas import (
    ATS,
    CandidateProfile,
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

TODAY = date(2026, 5, 27)


def _profile() -> CandidateProfile:
    return CandidateProfile(
        resume_text_sha256="a" * 64,
        summary="Senior life-sciences tech leader.",
        seniority_level=SeniorityLevel.SVP,
        industries=["life sciences", "pharma"],
        functional_expertise=["AI strategy", "transformation"],
        technical_expertise=["AI", "data", "cloud"],
        ai_data_cloud_experience=["GenAI", "data platforms"],
        transformation_themes=["AI transformation"],
        target_archetypes=["VP AI Transformation"],
        title_equivalents=["SVP Technology"],
    )


def _job(**overrides) -> ValidatedJob:
    url = "https://boards.greenhouse.io/acme/jobs/1"
    base = dict(
        raw_lead=RawJobLead(
            title_guess="VP AI",
            url=url,
            source_provider=SearchProviderName.TAVILY,
            source_query=SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text="x"),
        ),
        validation=ValidationResult(live=True, http_status=200, final_url=url),
        canonical_url=url,
        apply_url=url,
        dedup_hash="d" * 64,
        title="VP AI Transformation",
        company="Acme Bio",
        body_text=(
            "Lead AI transformation across pharma. We are urgently hiring "
            "for this role to drive impact across data platforms in life "
            "sciences. We have multiple openings on our senior leadership "
            "team. Contact jane@acme.com to learn more."
        ),
        location="Boston, MA",
        work_mode=WorkMode.REMOTE,
        ats=ATS.GREENHOUSE,
        freshness=Freshness(
            label=FreshnessLabel.RECENT,
            posted_at=date(2026, 5, 22),
            evidence=FreshnessEvidence(source="json_ld_datePosted", snippet="2026-05-22"),
        ),
        signals=JobSignals(
            urgency_phrases=["urgently hiring"],
            transformation_phrases=["ai transformation"],
            recruiter_contact="jane@acme.com",
            multiple_openings=True,
        ),
        status=JobStatus.ACTIVE,
    )
    base.update(overrides)
    return ValidatedJob(**base)


def _score(job, profile, criteria):
    return ScoringAgent(clock=lambda: TODAY).score(job, profile, criteria)


def _citation_for(bundle, key):
    return [c for c in bundle.citations if c.claim_key == key]


# ---------------------------------------------------------------------------
# Coverage invariant (the big one)
# ---------------------------------------------------------------------------

class TestCoverageInvariant:
    def test_every_nonzero_feature_key_is_cited(self):
        job = _job()
        profile = _profile()
        criteria = SearchCriteria()
        score = _score(job, profile, criteria)

        bundle = EvidenceAgent().extract(
            job=job, profile=profile, criteria=criteria, score=score
        )
        assert coverage_gaps(score, bundle) == []

    def test_zero_contribution_keys_not_cited(self):
        # Profile with industries the job doesn't mention -> industry feature 0.
        profile = _profile()
        profile = profile.model_copy(update={"industries": ["banking", "insurance"]})
        job = _job(body_text="A pure tech role about quantum computing. " * 5)
        criteria = SearchCriteria()
        score = _score(job, profile, criteria)

        bundle = EvidenceAgent().extract(
            job=job, profile=profile, criteria=criteria, score=score
        )
        # industry_overlap earned zero -> no citation for match.industry.
        assert _citation_for(bundle, "match.industry") == []


# ---------------------------------------------------------------------------
# Per-key citation handlers
# ---------------------------------------------------------------------------

class TestMatchCitations:
    def test_industry(self):
        bundle = EvidenceAgent().extract(
            job=_job(), profile=_profile(), criteria=SearchCriteria(),
            score=_score(_job(), _profile(), SearchCriteria()),
        )
        cites = _citation_for(bundle, "match.industry")
        assert cites
        # Quote contains 'life sciences' OR 'pharma' (whichever matched first).
        q = cites[0].quote.lower()
        assert "life sciences" in q or "pharma" in q

    def test_title(self):
        bundle = EvidenceAgent().extract(
            job=_job(), profile=_profile(), criteria=SearchCriteria(),
            score=_score(_job(), _profile(), SearchCriteria()),
        )
        cites = _citation_for(bundle, "match.title")
        assert cites
        assert cites[0].quote == "VP AI Transformation"

    def test_seniority(self):
        bundle = EvidenceAgent().extract(
            job=_job(), profile=_profile(), criteria=SearchCriteria(),
            score=_score(_job(), _profile(), SearchCriteria()),
        )
        cites = _citation_for(bundle, "match.seniority")
        # title is "VP AI Transformation" -> 'VP' phrase is the citable token.
        assert cites
        assert cites[0].quote.lower() == "vp"

    def test_location_remote(self):
        job = _job()
        crit = SearchCriteria(work_modes=[WorkMode.REMOTE], location_preference="Boston")
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=crit,
            score=_score(job, _profile(), crit),
        )
        cites = _citation_for(bundle, "match.location_remote")
        assert cites
        # Should cite the location string (Boston, MA).
        assert any("Boston" in c.quote for c in cites)

    def test_must_have_keyword(self):
        job = _job(body_text=_job().body_text + " GenAI is central to the role.")
        crit = SearchCriteria(must_have_keywords=["GenAI"])
        score = _score(job, _profile(), crit)
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=crit, score=score
        )
        cites = _citation_for(bundle, "match.keyword_must_have")
        assert cites
        assert "genai" in cites[0].quote.lower()


class TestUrgencyCitations:
    def test_posted_recently_cites_freshness_evidence(self):
        job = _job()
        score = _score(job, _profile(), SearchCriteria())
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=SearchCriteria(), score=score
        )
        cites = _citation_for(bundle, "urgency.posted_within_7d")
        assert cites
        assert cites[0].quote == "2026-05-22"

    def test_urgency_phrase(self):
        job = _job()
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=SearchCriteria(),
            score=_score(job, _profile(), SearchCriteria()),
        )
        cites = _citation_for(bundle, "urgency.phrase")
        assert cites
        assert "urgently hiring" in cites[0].quote.lower()

    def test_recruiter_contact_cited(self):
        job = _job()
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=SearchCriteria(),
            score=_score(job, _profile(), SearchCriteria()),
        )
        cites = _citation_for(bundle, "urgency.recruiter_contact")
        assert cites
        assert cites[0].quote == "jane@acme.com"

    def test_multiple_openings_cited_with_span(self):
        # Default fixture body already contains "multiple openings".
        job = _job()
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=SearchCriteria(),
            score=_score(job, _profile(), SearchCriteria()),
        )
        cites = _citation_for(bundle, "urgency.multiple_openings")
        assert cites
        assert "multiple openings" in cites[0].quote.lower()
        assert cites[0].start_idx is not None
        assert cites[0].end_idx is not None
        assert cites[0].end_idx > cites[0].start_idx

    def test_transformation_phrase(self):
        job = _job()
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=SearchCriteria(),
            score=_score(job, _profile(), SearchCriteria()),
        )
        cites = _citation_for(bundle, "urgency.transformation_phrase")
        assert cites
        assert "ai transformation" in cites[0].quote.lower()

    def test_source_ats_quotes_url(self):
        job = _job(ats=ATS.GREENHOUSE)
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=SearchCriteria(),
            score=_score(job, _profile(), SearchCriteria()),
        )
        cites = _citation_for(bundle, "urgency.source_ats")
        assert cites
        assert "greenhouse.io" in cites[0].quote


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdges:
    def test_empty_body_means_no_match_citations(self):
        # Both title AND body must avoid every profile needle. Pydantic also
        # requires body_text >= 20 chars.
        job = _job(
            title="Generic Position",
            body_text="apple banana cherry durian eggplant fig grape kiwi lemon",
            signals=JobSignals(),
        )
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=SearchCriteria(),
            score=_score(job, _profile(), SearchCriteria()),
        )
        # No content-derived match citations.
        for c in bundle.citations:
            assert c.claim_key not in (
                "match.industry",
                "match.functional_expertise",
                "match.tech_domain",
                "match.transformation",
            )

    def test_freshness_unknown_means_no_recency_citation(self):
        job = _job(
            freshness=Freshness(label=FreshnessLabel.UNKNOWN),
            signals=JobSignals(),
        )
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=SearchCriteria(),
            score=_score(job, _profile(), SearchCriteria()),
        )
        assert _citation_for(bundle, "urgency.posted_within_7d") == []
        assert _citation_for(bundle, "urgency.posted_within_14d") == []

    def test_keys_only_emitted_for_nonzero_features(self):
        job = _job()
        score = _score(job, _profile(), SearchCriteria())
        bundle = EvidenceAgent().extract(
            job=job, profile=_profile(), criteria=SearchCriteria(), score=score
        )
        # bundle.keys() should be a subset of what nonzero features claim.
        claimed: set[str] = set()
        for f in score.match_features + score.urgency_features:
            if f.contribution > 0:
                claimed.update(f.evidence_keys)
        for k in bundle.keys():
            assert k in claimed, f"bundle emitted unclaimed key {k!r}"
