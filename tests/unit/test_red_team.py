"""Red Team Agent unit tests.

Builds a happy-path report, then mutates one thing at a time to drive each
rejection rule. Each test asserts the SPECIFIC reason string so future
changes to the rule set produce loud test failures, not silent drift.
"""
from __future__ import annotations

from datetime import date

import pytest

from job_scout.agents.evidence import EvidenceAgent
from job_scout.agents.red_team import Reasons, RedTeamAgent
from job_scout.agents.report import ReportAgent
from job_scout.agents.scoring import ScoringAgent
from job_scout.schemas import (
    ATS,
    CandidateProfile,
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
            source_query=SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text='"VP AI"'),
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


def _assemble(job, criteria):
    profile = _profile()
    score = ScoringAgent(clock=lambda: TODAY).score(job, profile, criteria)
    evidence = EvidenceAgent().extract(
        job=job, profile=profile, criteria=criteria, score=score
    )
    return ReportAgent().assemble(job=job, score=score, evidence=evidence)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_strong_job_accepted(self):
        assembly = _assemble(_job(), SearchCriteria())
        decision = RedTeamAgent().evaluate(assembly, SearchCriteria())
        assert decision.accepted, decision.reasons
        assert decision.reasons == []


# ---------------------------------------------------------------------------
# Individual rejection rules
# ---------------------------------------------------------------------------

class TestRejectionRules:
    def test_below_min_match_threshold(self):
        # Strong job but high threshold
        assembly = _assemble(_job(), SearchCriteria(min_match_score=99))
        decision = RedTeamAgent().evaluate(assembly, SearchCriteria(min_match_score=99))
        assert not decision.accepted
        assert any(Reasons.MATCH_BELOW_THRESHOLD in r for r in decision.reasons)

    def test_older_posting_rejected_by_default(self):
        # OLDER label + criteria.allow_older=False (default).
        job = _job(
            freshness=Freshness(
                label=FreshnessLabel.OLDER,
                posted_at=date(2024, 1, 1),
                evidence=FreshnessEvidence(source="json_ld_datePosted", snippet="2024-01-01"),
            )
        )
        assembly = _assemble(job, SearchCriteria())
        decision = RedTeamAgent().evaluate(assembly, SearchCriteria())
        assert not decision.accepted
        assert any(Reasons.POSTING_TOO_OLD in r for r in decision.reasons)

    def test_older_posting_allowed_when_user_opts_in(self):
        job = _job(
            freshness=Freshness(
                label=FreshnessLabel.OLDER,
                posted_at=date(2024, 1, 1),
                evidence=FreshnessEvidence(source="json_ld_datePosted", snippet="2024-01-01"),
            )
        )
        criteria = SearchCriteria(allow_older=True, min_match_score=0)
        assembly = _assemble(job, criteria)
        decision = RedTeamAgent().evaluate(assembly, criteria)
        # Should not be rejected for being old.
        assert all(Reasons.POSTING_TOO_OLD not in r for r in decision.reasons)

    def test_unknown_freshness_not_rejected_for_age(self):
        # UNKNOWN is not stale by default - we don't know.
        job = _job(
            freshness=Freshness(label=FreshnessLabel.UNKNOWN),
            signals=JobSignals(),  # consistent: no signals
            body_text=(
                "Lead AI transformation across pharma. Drive impact across "
                "data platforms in life sciences. Senior leadership role."
            ),
        )
        criteria = SearchCriteria(min_match_score=0)
        assembly = _assemble(job, criteria)
        decision = RedTeamAgent().evaluate(assembly, criteria)
        assert all(Reasons.POSTING_TOO_OLD not in r for r in decision.reasons)

    def test_job_not_active_rejected(self):
        job = _job(status=JobStatus.CLOSED)
        assembly = _assemble(job, SearchCriteria())
        decision = RedTeamAgent().evaluate(assembly, SearchCriteria())
        assert not decision.accepted
        assert any(Reasons.JOB_NOT_ACTIVE in r for r in decision.reasons)

    def test_validation_not_live_rejected(self):
        job = _job(
            validation=ValidationResult(
                live=False,
                http_status=200,
                final_url="https://boards.greenhouse.io/acme/jobs/1",
                error=None,
            )
        )
        assembly = _assemble(job, SearchCriteria())
        decision = RedTeamAgent().evaluate(assembly, SearchCriteria())
        assert not decision.accepted
        assert Reasons.SOURCE_NOT_LIVE in decision.reasons

    def test_evidence_gap_rejected(self):
        # Construct a report with full scoring but empty evidence -> gap.
        job = _job()
        profile = _profile()
        criteria = SearchCriteria()
        score = ScoringAgent(clock=lambda: TODAY).score(job, profile, criteria)
        empty_evidence = EvidenceBundle(citations=[])
        assembly = ReportAgent().assemble(
            job=job, score=score, evidence=empty_evidence
        )
        decision = RedTeamAgent().evaluate(assembly, criteria)
        assert not decision.accepted
        assert any(r.startswith(Reasons.EVIDENCE_GAP) for r in decision.reasons)

    def test_duplicate_in_run_rejected(self):
        # Same job, evaluated twice in the same run.
        agent = RedTeamAgent()
        assembly = _assemble(_job(), SearchCriteria())
        d1 = agent.evaluate(assembly, SearchCriteria())
        d2 = agent.evaluate(assembly, SearchCriteria())
        assert d1.accepted
        assert not d2.accepted
        assert Reasons.DUPLICATE_IN_RUN in d2.reasons

    def test_duplicate_resets_on_reset(self):
        agent = RedTeamAgent()
        assembly = _assemble(_job(), SearchCriteria())
        agent.evaluate(assembly, SearchCriteria())
        agent.reset()
        d = agent.evaluate(assembly, SearchCriteria())
        assert d.accepted


# ---------------------------------------------------------------------------
# Multiple rejections at once
# ---------------------------------------------------------------------------

class TestMultipleRejections:
    def test_collects_all_failing_rules(self):
        # OLDER posting AND match below threshold AND not-active.
        job = _job(
            status=JobStatus.CLOSED,
            freshness=Freshness(
                label=FreshnessLabel.OLDER,
                posted_at=date(2024, 1, 1),
                evidence=FreshnessEvidence(source="json_ld_datePosted", snippet="2024-01-01"),
            ),
        )
        criteria = SearchCriteria(min_match_score=99)
        assembly = _assemble(job, criteria)
        decision = RedTeamAgent().evaluate(assembly, criteria)
        assert not decision.accepted
        # All three rules trigger.
        assert any(Reasons.POSTING_TOO_OLD in r for r in decision.reasons)
        assert any(Reasons.JOB_NOT_ACTIVE in r for r in decision.reasons)
        assert any(Reasons.MATCH_BELOW_THRESHOLD in r for r in decision.reasons)
