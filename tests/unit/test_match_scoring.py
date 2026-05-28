"""Match score feature tests.

Builds canonical strong/weak/misleading scenarios and pins the contribution
ranges. Also confirms the invariant that every non-zero contribution carries
at least one evidence key.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.schemas import (
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
from src.scoring.match import compute_match_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _profile(**overrides) -> CandidateProfile:
    base = dict(
        resume_text_sha256="a" * 64,
        summary="Senior life-sciences tech leader.",
        seniority_level=SeniorityLevel.SVP,
        industries=["life sciences", "pharma"],
        functional_expertise=["AI strategy", "transformation", "P&L"],
        technical_expertise=["AI", "data", "cloud"],
        ai_data_cloud_experience=["GenAI", "data platforms"],
        transformation_themes=["AI transformation", "digital reinvention"],
        target_archetypes=["VP AI Transformation", "Chief Digital Officer"],
        title_equivalents=["SVP Technology", "Head of AI"],
        search_keywords=["AI transformation"],
        ranking_keywords=["AI", "pharma"],
    )
    base.update(overrides)
    return CandidateProfile(**base)


def _job(
    *,
    title: str = "VP AI Transformation",
    company: str = "Acme Bio",
    body_text: str = (
        "Lead AI transformation across our pharma business. "
        "Drive measurable impact across data platforms in life sciences. "
        "Senior leadership role with P&L responsibility."
    ),
    work_mode: WorkMode = WorkMode.REMOTE,
    location: str | None = "Boston, MA",
    ats: ATS | None = ATS.GREENHOUSE,
    posted_at: date | None = date(2026, 5, 20),
) -> ValidatedJob:
    url = "https://boards.greenhouse.io/acme/jobs/1"
    return ValidatedJob(
        raw_lead=RawJobLead(
            title_guess=title,
            url=url,
            source_provider=SearchProviderName.TAVILY,
            source_query=SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text='"VP AI"'),
        ),
        validation=ValidationResult(live=True, http_status=200, final_url=url),
        canonical_url=url,
        apply_url=url,
        dedup_hash="d" * 64,
        title=title,
        company=company,
        body_text=body_text,
        location=location,
        work_mode=work_mode,
        ats=ats,
        freshness=Freshness(
            label=FreshnessLabel.RECENT if posted_at else FreshnessLabel.UNKNOWN,
            posted_at=posted_at,
            evidence=(
                FreshnessEvidence(source="json_ld_datePosted", snippet=str(posted_at))
                if posted_at
                else None
            ),
        ),
        signals=JobSignals(),
        status=JobStatus.ACTIVE,
    )


def _feature(features, name):
    for f in features:
        if f.name == name:
            return f
    raise AssertionError(f"feature {name!r} not found")


# ---------------------------------------------------------------------------
# Total weight invariant
# ---------------------------------------------------------------------------

class TestTotalWeight:
    def test_weights_sum_to_100(self):
        features = compute_match_features(_job(), _profile(), SearchCriteria())
        assert sum(f.weight for f in features) == 100


# ---------------------------------------------------------------------------
# Strong / weak / misleading scenarios
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_strong_match_scores_high(self):
        features = compute_match_features(_job(), _profile(), SearchCriteria())
        total = sum(f.contribution for f in features)
        assert total >= 70

    def test_weak_match_scores_low(self):
        # Profile for life-sciences SVP; job is a junior frontend dev at a fintech.
        job = _job(
            title="Junior Frontend Developer",
            company="FintechCo",
            body_text=(
                "We need a junior React developer to build dashboards. "
                "Bootcamp grads welcome. No leadership experience required."
            ),
            ats=None,
            location="San Francisco, CA",
        )
        features = compute_match_features(job, _profile(), SearchCriteria())
        total = sum(f.contribution for f in features)
        assert total <= 25, f"weak match scored {total}"

    def test_misleading_title_still_scores_low_on_body(self):
        # Title contains 'VP AI' but body is irrelevant junior work.
        job = _job(
            title="VP AI Marketing Coordinator",
            body_text=(
                "Manage marketing emails and run AI-themed campaigns. "
                "Entry-level position, no leadership."
            ),
            ats=None,
        )
        features = compute_match_features(job, _profile(), SearchCriteria())
        # Title may earn 'title_match', but industry/functional should be low.
        ind = _feature(features, "industry_overlap")
        func = _feature(features, "functional_expertise_overlap")
        # These should not have full credit since the body is irrelevant.
        # 'AI' is in tech_domain so that may earn some, but functional expertise
        # words (P&L, transformation, AI strategy) are not in the marketing body.
        assert func.contribution < func.weight * 0.7


# ---------------------------------------------------------------------------
# Evidence keys
# ---------------------------------------------------------------------------

class TestEvidenceKeys:
    # `keyword_must_have` and `keyword_exclusion` are 'default-pass' features:
    # if the user didn't configure any constraint, they award full credit
    # without a citation (there's nothing to cite). All other features that
    # earn points MUST carry an evidence key.
    _DEFAULT_PASS_NAMES = {"keyword_must_have", "keyword_exclusion"}

    def test_nonzero_content_contributions_have_evidence_keys(self):
        features = compute_match_features(_job(), _profile(), SearchCriteria())
        for f in features:
            if f.contribution > 0 and f.name not in self._DEFAULT_PASS_NAMES:
                assert f.evidence_keys, (
                    f"feature {f.name} has contribution {f.contribution} "
                    f"but no evidence keys (notes={f.notes!r})"
                )

    def test_default_pass_features_award_credit_without_evidence(self):
        # SearchCriteria() has no must_have_keywords and no exclusion_keywords.
        features = compute_match_features(_job(), _profile(), SearchCriteria())
        for name in self._DEFAULT_PASS_NAMES:
            f = _feature(features, name)
            assert f.contribution == f.weight
            assert f.evidence_keys == []
            assert "no" in f.notes.lower() and ("configured" in f.notes or "to check" in f.notes)

    def test_zero_contributions_have_no_evidence_keys(self):
        # Empty profile lists -> overlap features all zero with no evidence.
        empty = _profile(
            industries=[],
            functional_expertise=[],
            technical_expertise=[],
            ai_data_cloud_experience=[],
            transformation_themes=[],
            target_archetypes=[],
            title_equivalents=[],
        )
        features = compute_match_features(_job(), empty, SearchCriteria())
        for f in features:
            if f.contribution == 0:
                assert f.evidence_keys == [], (
                    f"feature {f.name} has zero contribution but evidence_keys={f.evidence_keys}"
                )


# ---------------------------------------------------------------------------
# Individual features
# ---------------------------------------------------------------------------

class TestIndividualFeatures:
    def test_title_match_exact(self):
        job = _job(title="VP AI Transformation")
        f = _feature(compute_match_features(job, _profile(), SearchCriteria()), "title_match")
        assert f.contribution == f.weight
        assert "match.title" in f.evidence_keys

    def test_title_match_zero_when_unrelated(self):
        job = _job(title="Marketing Coordinator")
        f = _feature(compute_match_features(job, _profile(), SearchCriteria()), "title_match")
        assert f.contribution == 0.0
        assert f.evidence_keys == []

    def test_seniority_match_exact(self):
        job = _job(title="SVP Technology")  # SVP -> rank 4, profile SVP -> rank 4
        f = _feature(compute_match_features(job, _profile(), SearchCriteria()), "seniority_match")
        assert f.contribution == f.weight

    def test_seniority_match_partial(self):
        job = _job(title="VP Technology")  # VP -> rank 3, profile SVP -> rank 4, delta 1
        f = _feature(compute_match_features(job, _profile(), SearchCriteria()), "seniority_match")
        # delta 1 -> 70% credit
        assert f.contribution == pytest.approx(f.weight * 0.7)

    def test_seniority_undetected_yields_zero(self):
        job = _job(title="Random Title Without Seniority Cue")
        f = _feature(compute_match_features(job, _profile(), SearchCriteria()), "seniority_match")
        assert f.contribution == 0.0

    def test_location_remote_matches_criteria(self):
        job = _job(work_mode=WorkMode.REMOTE, location="Boston, MA")
        crit = SearchCriteria(work_modes=[WorkMode.REMOTE], location_preference="Boston")
        f = _feature(compute_match_features(job, _profile(), crit), "location_remote_fit")
        assert f.contribution == f.weight

    def test_location_mismatch_drops_to_partial(self):
        job = _job(work_mode=WorkMode.ONSITE, location="London")
        crit = SearchCriteria(work_modes=[WorkMode.REMOTE], location_preference="Boston")
        f = _feature(compute_match_features(job, _profile(), crit), "location_remote_fit")
        assert f.contribution == 0.0  # both subscores zero

    def test_exclusion_keyword_zeros_the_feature(self):
        job = _job(body_text="This role is at a defense contractor. " * 5 + "We build weapons.")
        crit = SearchCriteria(exclusion_keywords=["defense", "weapons"])
        f = _feature(compute_match_features(job, _profile(), crit), "keyword_exclusion")
        assert f.contribution == 0.0
        assert f.evidence_keys == []

    def test_must_have_keyword_present(self):
        job = _job(body_text=_job().body_text + " GenAI is core to our work.")
        crit = SearchCriteria(must_have_keywords=["GenAI"])
        f = _feature(compute_match_features(job, _profile(), crit), "keyword_must_have")
        assert f.contribution == f.weight
        assert "match.keyword_must_have" in f.evidence_keys

    def test_must_have_keyword_missing(self):
        job = _job(body_text="Lead AI transformation across pharma. Bog standard.")
        crit = SearchCriteria(must_have_keywords=["Quantum Computing"])
        f = _feature(compute_match_features(job, _profile(), crit), "keyword_must_have")
        assert f.contribution == 0.0
        assert f.evidence_keys == []
