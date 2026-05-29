"""In-memory repository implementations.

Used by unit tests and as a graceful fallback when Supabase is unreachable.
NOT for production data. Thread-unsafe by design (tests are single-threaded
and the Streamlit fallback is single-session).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from job_scout.schemas import (
    CandidateProfile,
    JobReport,
    JobStatus,
    RunRecord,
    UserActions,
    ValidatedJob,
)

# Internal storage models (job_id <-> ValidatedJob).
# We keep the assigned UUID alongside the schema because ValidatedJob does
# not itself carry an id — ids are an artifact of persistence.


@dataclass
class _ProfileStore:
    by_id: dict[UUID, CandidateProfile] = field(default_factory=dict)
    by_hash: dict[str, UUID] = field(default_factory=dict)


@dataclass
class _JobStore:
    by_id: dict[UUID, ValidatedJob] = field(default_factory=dict)
    by_hash: dict[str, UUID] = field(default_factory=dict)


@dataclass
class _RunStore:
    by_id: dict[UUID, RunRecord] = field(default_factory=dict)


@dataclass
class _UserActionStore:
    by_pair: dict[tuple[UUID, UUID], UserActions] = field(default_factory=dict)


class InMemoryProfileRepo:
    def __init__(self, store: _ProfileStore) -> None:
        self._s = store

    def upsert(self, profile: CandidateProfile) -> UUID:
        # If we've seen this resume text before, return the existing id.
        existing = self._s.by_hash.get(profile.resume_text_sha256)
        if existing is not None:
            self._s.by_id[existing] = profile
            return existing
        new_id = uuid4()
        self._s.by_id[new_id] = profile
        self._s.by_hash[profile.resume_text_sha256] = new_id
        return new_id

    def get(self, profile_id: UUID) -> Optional[CandidateProfile]:
        return self._s.by_id.get(profile_id)

    def find_by_resume_hash(self, sha256: str) -> Optional[UUID]:
        return self._s.by_hash.get(sha256)


class InMemoryJobRepo:
    def __init__(self, store: _JobStore) -> None:
        self._s = store

    def upsert(self, job: ValidatedJob) -> UUID:
        existing = self._s.by_hash.get(job.dedup_hash)
        if existing is not None:
            self._s.by_id[existing] = job
            return existing
        new_id = uuid4()
        self._s.by_id[new_id] = job
        self._s.by_hash[job.dedup_hash] = new_id
        return new_id

    def get(self, job_id: UUID) -> Optional[ValidatedJob]:
        return self._s.by_id.get(job_id)

    def find_by_dedup_hash(self, dedup_hash: str) -> Optional[UUID]:
        return self._s.by_hash.get(dedup_hash)

    def mark_status(self, job_id: UUID, status: str) -> None:
        # `model_copy(update=...)` does not re-validate fields, so coerce
        # the incoming string to the JobStatus enum here. This keeps the
        # in-memory store byte-for-byte consistent with the schema.
        job = self._s.by_id.get(job_id)
        if job is None:
            return
        self._s.by_id[job_id] = job.model_copy(update={"status": JobStatus(status)})


class InMemoryRunRepo:
    def __init__(self, store: _RunStore) -> None:
        self._s = store

    def insert(self, run: RunRecord) -> UUID:
        self._s.by_id[run.id] = run
        return run.id

    def get(self, run_id: UUID) -> Optional[RunRecord]:
        return self._s.by_id.get(run_id)

    def attach_report(self, run_id: UUID, job_id: UUID, report: JobReport) -> None:
        run = self._s.by_id.get(run_id)
        if run is None:
            return
        # Replace any prior report for this job_id (rare but possible during reruns).
        # We use raw_lead.url as a stand-in identity since reports don't carry job_id.
        existing_idx = None
        for i, r in enumerate(run.reports):
            if r.job.dedup_hash == report.job.dedup_hash:
                existing_idx = i
                break
        updated_reports = list(run.reports)
        if existing_idx is None:
            updated_reports.append(report)
        else:
            updated_reports[existing_idx] = report
        self._s.by_id[run_id] = run.model_copy(update={"reports": updated_reports})


class InMemoryUserActionsRepo:
    def __init__(self, store: _UserActionStore) -> None:
        self._s = store

    def get(self, profile_id: UUID, job_id: UUID) -> Optional[UserActions]:
        return self._s.by_pair.get((profile_id, job_id))

    def upsert(self, actions: UserActions) -> None:
        self._s.by_pair[(actions.profile_id, actions.job_id)] = actions

    def list_for_profile(self, profile_id: UUID) -> list[UserActions]:
        return [a for (pid, _), a in self._s.by_pair.items() if pid == profile_id]


class InMemoryRepositories:
    """Bundle satisfying `Repositories` for tests + Supabase fallback."""

    def __init__(self) -> None:
        self._profile_store = _ProfileStore()
        self._job_store = _JobStore()
        self._run_store = _RunStore()
        self._user_action_store = _UserActionStore()
        self.profiles: InMemoryProfileRepo = InMemoryProfileRepo(self._profile_store)
        self.jobs: InMemoryJobRepo = InMemoryJobRepo(self._job_store)
        self.runs: InMemoryRunRepo = InMemoryRunRepo(self._run_store)
        self.user_actions: InMemoryUserActionsRepo = InMemoryUserActionsRepo(
            self._user_action_store
        )
