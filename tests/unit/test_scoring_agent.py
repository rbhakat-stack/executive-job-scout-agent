"""ScoringAgent end-to-end tests, including the LLM rationale fallback."""
from __future__ import annotations

import json
from datetime import date

from src.agents.scoring import ScoringAgent
from src.llm import FakeLLM
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
from src.scoring.rationale import is_substantive_rationale
from src.scoring.match import compute_match_features

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
            "Lead AI transformation across pharma. Drive impact across "
            "data platforms in life sciences. Senior leadership role."
        ),
        location="Boston, MA",
        work_mode=WorkMode.REMOTE,
        ats=ATS.GREENHOUSE,
        freshness=Freshness(
            label=FreshnessLabel.RECENT,
            posted_at=date(2026, 5, 22),
            evidence=FreshnessEvidence(source="json_ld_datePosted", snippet="2026-05-22"),
        ),
        signals=JobSignals(),
        status=JobStatus.ACTIVE,
    )
    base.update(overrides)
    return ValidatedJob(**base)


# ---------------------------------------------------------------------------
# Without LLM (deterministic floor only)
# ---------------------------------------------------------------------------

class TestDeterministicOnly:
    def test_returns_valid_score_result(self):
        result = ScoringAgent(clock=lambda: TODAY).score(
            _job(), _profile(), SearchCriteria()
        )
        assert 0 <= result.match_score <= 100
        assert 0 <= result.urgency_score <= 100
        # Substantive: mentions job title or company AND a feature.
        assert is_substantive_rationale(
            result.match_rationale,
            job=_job(),
            match_features=result.match_features,
        )

    def test_strong_match_high_score(self):
        result = ScoringAgent(clock=lambda: TODAY).score(
            _job(), _profile(), SearchCriteria()
        )
        assert result.match_score >= 70

    def test_application_and_outreach_angles_populated(self):
        result = ScoringAgent(clock=lambda: TODAY).score(
            _job(), _profile(), SearchCriteria()
        )
        assert result.application_angle and len(result.application_angle) > 10
        assert result.outreach_angle and len(result.outreach_angle) > 10

    def test_urgency_features_picked_up(self):
        job = _job(
            signals=JobSignals(
                urgency_phrases=["urgently hiring"],
                multiple_openings=True,
                transformation_phrases=["ai transformation"],
            )
        )
        result = ScoringAgent(clock=lambda: TODAY).score(job, _profile(), SearchCriteria())
        # 25 + 15 + 15 + 10 + 10 + 15 = 90 max if recruiter contact missing
        assert result.urgency_score >= 80


# ---------------------------------------------------------------------------
# With LLM
# ---------------------------------------------------------------------------

class TestLLMRationale:
    def test_good_llm_response_is_used(self):
        good = {
            "rationale": (
                "Acme Bio's VP AI Transformation role aligns to industry overlap "
                "and tech domain overlap from your profile (match.industry, "
                "match.tech_domain)."
            ),
            "concerns": "Limited info on team size.",
            "application_angle": "Lead with pharma platform reinvention experience.",
            "outreach_angle": "Reference your AI-led commercial program at Acme Pharma.",
        }
        llm = FakeLLM(responses=[json.dumps(good)])
        result = ScoringAgent(llm, clock=lambda: TODAY).score(
            _job(), _profile(), SearchCriteria()
        )
        assert result.match_rationale.startswith("Acme Bio")
        assert result.concerns == "Limited info on team size."
        assert "pharma platform" in result.application_angle

    def test_generic_llm_response_falls_back_to_deterministic(self):
        generic = {
            "rationale": "Strong fit.",  # too short + no specifics
            "concerns": None,
            "application_angle": "Apply.",
            "outreach_angle": "Reach out.",
        }
        llm = FakeLLM(responses=[json.dumps(generic)])
        agent = ScoringAgent(llm, clock=lambda: TODAY)
        result = agent.score(_job(), _profile(), SearchCriteria())
        # The deterministic floor mentions specifics like the score and company.
        assert "Acme Bio" in result.match_rationale
        assert "Match score" in result.match_rationale

    def test_malformed_llm_response_falls_back(self):
        llm = FakeLLM(responses=["this is not json at all"])
        result = ScoringAgent(llm, clock=lambda: TODAY).score(
            _job(), _profile(), SearchCriteria()
        )
        # Still produces a usable rationale via the deterministic path.
        assert is_substantive_rationale(
            result.match_rationale,
            job=_job(),
            match_features=result.match_features,
        )

    def test_llm_rationale_without_feature_keys_falls_back(self):
        # The rationale mentions the company but does NOT reference any
        # feature name or evidence key -> failing the substantive check.
        obj = {
            "rationale": "Acme Bio is a great place to work. Lots of growth.",
            "concerns": None,
            "application_angle": "Apply with enthusiasm.",
            "outreach_angle": "Be brief.",
        }
        llm = FakeLLM(responses=[json.dumps(obj)])
        result = ScoringAgent(llm, clock=lambda: TODAY).score(
            _job(), _profile(), SearchCriteria()
        )
        # Deterministic floor kicks in.
        assert "Match score" in result.match_rationale


# ---------------------------------------------------------------------------
# Substantive-rationale heuristic
# ---------------------------------------------------------------------------

class TestSkipLLM:
    """The `skip_llm=True` path forces deterministic-only scoring even
    when an LLM is configured. Used by the orchestrator's two-pass flow
    to avoid burning tokens (and hitting rate limits) on jobs that will
    be filtered by the match-score threshold anyway.
    """

    def test_skip_llm_does_not_call_the_llm(self):
        # FakeLLM with an empty queue would raise on .complete() — proves
        # the agent didn't call it.
        llm = FakeLLM(responses=[])
        agent = ScoringAgent(llm, clock=lambda: TODAY)
        result = agent.score(_job(), _profile(), SearchCriteria(), skip_llm=True)
        assert llm.calls == []
        # Rationale still populated via the deterministic floor.
        assert "Match score" in result.match_rationale

    def test_default_is_to_use_llm(self):
        llm = FakeLLM(
            responses=[
                json.dumps(
                    {
                        "rationale": (
                            "Acme Bio aligns to industry_overlap and "
                            "match.industry citations."
                        ),
                        "concerns": None,
                        "application_angle": "Apply.",
                        "outreach_angle": "Reach out.",
                    }
                )
            ]
        )
        agent = ScoringAgent(llm, clock=lambda: TODAY)
        agent.score(_job(), _profile(), SearchCriteria())
        assert len(llm.calls) == 1


class TestSubstantiveCheck:
    def test_short_rationale_rejected(self):
        assert not is_substantive_rationale(
            "great fit",
            job=_job(),
            match_features=compute_match_features(_job(), _profile(), SearchCriteria()),
        )

    def test_no_company_or_title_token_rejected(self):
        assert not is_substantive_rationale(
            "This is a wonderful opportunity that aligns with everything. " * 2,
            job=_job(),
            match_features=compute_match_features(_job(), _profile(), SearchCriteria()),
        )

    def test_with_company_and_feature_accepted(self):
        assert is_substantive_rationale(
            "Acme Bio aligns strongly to your industry_overlap and "
            "title_match features. A clear hit.",
            job=_job(),
            match_features=compute_match_features(_job(), _profile(), SearchCriteria()),
        )
