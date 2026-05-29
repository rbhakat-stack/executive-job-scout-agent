"""Behavior tests for the in-memory repository.

These pin the persistence contract that the Supabase implementation must
also satisfy:
- profiles dedup by resume_text_sha256
- jobs dedup by dedup_hash
- runs.attach_report replaces rather than duplicating per-job entries
- user_actions is keyed by (profile_id, job_id)
"""
from __future__ import annotations

from datetime import date
from uuid import uuid4

from job_scout.db.memory_repo import InMemoryRepositories
from job_scout.schemas import (
    ATS,
    CandidateProfile,
    Citation,
    EvidenceBundle,
    Freshness,
    FreshnessEvidence,
    FreshnessLabel,
    JobReport,
    RawJobLead,
    RunRecord,
    ScoreResult,
    SearchCriteria,
    SearchProviderName,
    SearchQuery,
    SearchStrategy,
    SeniorityLevel,
    UserActions,
    ValidatedJob,
    ValidationResult,
)


def _profile(hash_seed: str = "a") -> CandidateProfile:
    return CandidateProfile(
        resume_text_sha256=hash_seed * 64,
        summary="Senior life-sciences tech leader.",
        seniority_level=SeniorityLevel.SVP,
    )


def _job(dedup: str = "d", url: str = "https://boards.greenhouse.io/acme/jobs/1") -> ValidatedJob:
    return ValidatedJob(
        raw_lead=RawJobLead(
            title_guess="VP AI",
            url=url,
            source_provider=SearchProviderName.TAVILY,
            source_query=SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text='"VP AI"'),
            ats_guess=ATS.GREENHOUSE,
        ),
        validation=ValidationResult(live=True, http_status=200, final_url=url),
        canonical_url=url,
        apply_url=url + "/apply",
        dedup_hash=dedup * 64,
        title="VP AI",
        company="Acme",
        body_text="A substantive job description that easily clears the minimum length.",
        freshness=Freshness(
            label=FreshnessLabel.RECENT,
            posted_at=date(2026, 5, 20),
            evidence=FreshnessEvidence(source="ats_field", snippet="2026-05-20"),
        ),
    )


def _report(job: ValidatedJob) -> JobReport:
    return JobReport(
        job=job,
        score=ScoreResult(
            match_score=85,
            urgency_score=70,
            match_rationale=(
                "Strong industry overlap (life sciences) and seniority match per "
                "match.industry citation."
            ),
        ),
        evidence=EvidenceBundle(
            citations=[
                Citation(
                    claim_key="match.industry",
                    quote="life sciences",
                    source_url=job.canonical_url,
                )
            ]
        ),
    )


class TestProfileRepo:
    def test_upsert_returns_same_id_for_same_hash(self):
        repo = InMemoryRepositories()
        p = _profile()
        id1 = repo.profiles.upsert(p)
        id2 = repo.profiles.upsert(p)
        assert id1 == id2

    def test_get_round_trips(self):
        repo = InMemoryRepositories()
        pid = repo.profiles.upsert(_profile())
        loaded = repo.profiles.get(pid)
        assert loaded is not None
        assert loaded.seniority_level is SeniorityLevel.SVP

    def test_find_by_resume_hash(self):
        repo = InMemoryRepositories()
        p = _profile("b")
        pid = repo.profiles.upsert(p)
        assert repo.profiles.find_by_resume_hash(p.resume_text_sha256) == pid
        assert repo.profiles.find_by_resume_hash("nope" * 16) is None


class TestJobRepo:
    def test_dedup_by_hash(self):
        repo = InMemoryRepositories()
        j1 = _job(dedup="d")
        j2 = _job(dedup="d", url="https://boards.greenhouse.io/acme/jobs/1?utm=x")
        id1 = repo.jobs.upsert(j1)
        id2 = repo.jobs.upsert(j2)
        assert id1 == id2

    def test_distinct_hashes_get_distinct_ids(self):
        repo = InMemoryRepositories()
        id1 = repo.jobs.upsert(_job(dedup="d"))
        id2 = repo.jobs.upsert(_job(dedup="e", url="https://boards.greenhouse.io/acme/jobs/2"))
        assert id1 != id2

    def test_mark_status(self):
        repo = InMemoryRepositories()
        jid = repo.jobs.upsert(_job())
        repo.jobs.mark_status(jid, "closed")
        assert repo.jobs.get(jid).status.value == "closed"


class TestRunRepo:
    def test_attach_report_idempotent_per_dedup_hash(self):
        repo = InMemoryRepositories()
        job = _job()
        run = RunRecord(criteria=SearchCriteria())
        rid = repo.runs.insert(run)
        repo.runs.attach_report(rid, uuid4(), _report(job))
        repo.runs.attach_report(rid, uuid4(), _report(job))  # same dedup_hash
        loaded = repo.runs.get(rid)
        assert len(loaded.reports) == 1


class TestUserActionsRepo:
    def test_upsert_and_get(self):
        repo = InMemoryRepositories()
        pid, jid = uuid4(), uuid4()
        ua = UserActions(profile_id=pid, job_id=jid, favorited=True, notes="follow up")
        repo.user_actions.upsert(ua)
        loaded = repo.user_actions.get(pid, jid)
        assert loaded.favorited is True
        assert loaded.notes == "follow up"

    def test_list_for_profile(self):
        repo = InMemoryRepositories()
        pid = uuid4()
        other = uuid4()
        repo.user_actions.upsert(UserActions(profile_id=pid, job_id=uuid4()))
        repo.user_actions.upsert(UserActions(profile_id=pid, job_id=uuid4()))
        repo.user_actions.upsert(UserActions(profile_id=other, job_id=uuid4()))
        assert len(repo.user_actions.list_for_profile(pid)) == 2
        assert len(repo.user_actions.list_for_profile(other)) == 1
