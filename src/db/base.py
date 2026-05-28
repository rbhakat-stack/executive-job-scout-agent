"""Repository interfaces.

The application code talks ONLY to these protocols. Concrete implementations
(Supabase, in-memory for tests) live alongside this module.

Design rules:
- Repos never throw on "not found"; they return None.
- Repos accept and return Pydantic schemas (`src.schemas.*`), never raw dicts.
- Repos do not validate business rules — they persist what the agents produce.
- API keys never enter a repo argument (enforced by the type system: there is
  no `api_key` field in any schema).
"""
from __future__ import annotations

from typing import Optional, Protocol
from uuid import UUID

from src.schemas import (
    CandidateProfile,
    JobReport,
    RunRecord,
    UserActions,
    ValidatedJob,
)


class ProfileRepo(Protocol):
    def upsert(self, profile: CandidateProfile) -> UUID: ...
    def get(self, profile_id: UUID) -> Optional[CandidateProfile]: ...
    def find_by_resume_hash(self, sha256: str) -> Optional[UUID]: ...


class JobRepo(Protocol):
    def upsert(self, job: ValidatedJob) -> UUID:
        """Insert or update by `dedup_hash`. Returns the canonical job id."""
        ...

    def get(self, job_id: UUID) -> Optional[ValidatedJob]: ...
    def find_by_dedup_hash(self, dedup_hash: str) -> Optional[UUID]: ...
    def mark_status(self, job_id: UUID, status: str) -> None: ...


class RunRepo(Protocol):
    def insert(self, run: RunRecord) -> UUID: ...
    def get(self, run_id: UUID) -> Optional[RunRecord]: ...
    def attach_report(self, run_id: UUID, job_id: UUID, report: JobReport) -> None: ...


class UserActionsRepo(Protocol):
    def get(self, profile_id: UUID, job_id: UUID) -> Optional[UserActions]: ...
    def upsert(self, actions: UserActions) -> None: ...
    def list_for_profile(self, profile_id: UUID) -> list[UserActions]: ...


class Repositories(Protocol):
    """Bundle handed to agents that need persistence."""

    profiles: ProfileRepo
    jobs: JobRepo
    runs: RunRepo
    user_actions: UserActionsRepo
