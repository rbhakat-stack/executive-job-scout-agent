"""Report Agent tests.

The Report Agent is structural — these tests pin that it assembles a
JobReport and surfaces any coverage gaps for the Red Team to act on.
"""
from __future__ import annotations

from datetime import date

from job_scout.agents.evidence import EvidenceAgent
from job_scout.agents.report import ReportAgent
from job_scout.agents.scoring import ScoringAgent
from job_scout.schemas import (
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

TODAY = date(2026, 5, 27)


def _job() -> ValidatedJob:
    url = "https://boards.greenhouse.io/acme/jobs/1"
    return ValidatedJob(
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
            "data platforms in life sciences."
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


def _profile() -> CandidateProfile:
    return CandidateProfile(
        resume_text_sha256="a" * 64,
        summary="Senior life-sciences tech leader.",
        seniority_level=SeniorityLevel.SVP,
        industries=["life sciences", "pharma"],
        functional_expertise=["AI strategy"],
        technical_expertise=["AI", "data"],
        ai_data_cloud_experience=["data platforms"],
        transformation_themes=["AI transformation"],
        target_archetypes=["VP AI Transformation"],
    )


class TestReportAssembly:
    def test_no_coverage_gaps_on_happy_path(self):
        job = _job()
        profile = _profile()
        criteria = SearchCriteria()
        score = ScoringAgent(clock=lambda: TODAY).score(job, profile, criteria)
        evidence = EvidenceAgent().extract(
            job=job, profile=profile, criteria=criteria, score=score
        )
        assembly = ReportAgent().assemble(job=job, score=score, evidence=evidence)

        assert assembly.report.job is job
        assert assembly.report.score is score
        assert assembly.report.evidence is evidence
        assert assembly.coverage_gaps == []

    def test_coverage_gaps_surfaced_when_evidence_is_empty(self):
        # Manually pair a real score with an empty evidence bundle - the
        # gaps list should expose every nonzero feature's evidence keys.
        job = _job()
        profile = _profile()
        criteria = SearchCriteria()
        score = ScoringAgent(clock=lambda: TODAY).score(job, profile, criteria)
        empty = EvidenceBundle(citations=[])

        assembly = ReportAgent().assemble(job=job, score=score, evidence=empty)
        # Every key that earned a non-zero contribution should be in gaps.
        for f in score.match_features + score.urgency_features:
            if f.contribution > 0:
                for k in f.evidence_keys:
                    assert k in assembly.coverage_gaps

    def test_partial_evidence_only_lists_missing_keys(self):
        job = _job()
        profile = _profile()
        criteria = SearchCriteria()
        score = ScoringAgent(clock=lambda: TODAY).score(job, profile, criteria)
        # Provide ONE citation, for match.title only.
        partial = EvidenceBundle(
            citations=[
                Citation(
                    claim_key="match.title",
                    quote="VP AI Transformation",
                    source_url=job.canonical_url,
                )
            ]
        )
        assembly = ReportAgent().assemble(job=job, score=score, evidence=partial)
        assert "match.title" not in assembly.coverage_gaps
        # Other keys still gap.
        assert any(g.startswith("match.") for g in assembly.coverage_gaps)
