"""Supabase repository implementation.

Thin adapter over `supabase-py`. The `supabase` client is imported lazily so
that test environments and local runs without Supabase configured do not
need the dependency at import time.

Notes:
- We persist Pydantic schemas via `model_dump(mode='json')` so enums become
  strings, datetimes become ISO 8601, and UUIDs become canonical strings.
- We do not catch generic exceptions; the caller (orchestrator) is responsible
  for the "Supabase down -> fall back to InMemoryRepositories" path.
"""
from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from src.schemas import (
    CandidateProfile,
    JobReport,
    RunRecord,
    UserActions,
    ValidatedJob,
)


def _client(url: str, key: str):
    """Lazy import so tests/local runs without supabase configured don't pay for it."""
    from supabase import create_client  # type: ignore

    return create_client(url, key)


def _dump(model) -> dict[str, Any]:
    return model.model_dump(mode="json")


class SupabaseProfileRepo:
    def __init__(self, client) -> None:
        self._c = client

    def upsert(self, profile: CandidateProfile) -> UUID:
        row = {
            "resume_filename": profile.resume_filename,
            "resume_text_sha256": profile.resume_text_sha256,
            # NOTE: resume_text itself is set by the caller before persistence,
            # since the schema only carries the hash. We keep this adapter
            # 1-to-1 with schema fields; callers writing raw resume text are
            # expected to populate it via a separate path (see orchestrator).
            "resume_text": "",
            "linkedin_url": profile.linkedin_url,
            "extracted": _dump(profile),
            "seniority_level": profile.seniority_level.value,
            "industries": profile.industries,
            "target_titles": profile.target_archetypes,
        }
        existing = self.find_by_resume_hash(profile.resume_text_sha256)
        if existing is not None:
            self._c.table("profiles").update(row).eq("id", str(existing)).execute()
            return existing
        res = self._c.table("profiles").insert(row).execute()
        return UUID(res.data[0]["id"])

    def get(self, profile_id: UUID) -> Optional[CandidateProfile]:
        res = (
            self._c.table("profiles")
            .select("extracted")
            .eq("id", str(profile_id))
            .limit(1)
            .execute()
        )
        if not res.data:
            return None
        return CandidateProfile.model_validate(res.data[0]["extracted"])

    def find_by_resume_hash(self, sha256: str) -> Optional[UUID]:
        res = (
            self._c.table("profiles")
            .select("id")
            .eq("resume_text_sha256", sha256)
            .limit(1)
            .execute()
        )
        if not res.data:
            return None
        return UUID(res.data[0]["id"])


class SupabaseJobRepo:
    def __init__(self, client) -> None:
        self._c = client

    def upsert(self, job: ValidatedJob) -> UUID:
        row = {
            "dedup_hash": job.dedup_hash,
            "canonical_url": str(job.canonical_url),
            "apply_url": str(job.apply_url),
            "source_url": str(job.raw_lead.url),
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "work_mode": job.work_mode.value,
            "posted_at": job.freshness.posted_at.isoformat()
            if job.freshness.posted_at
            else None,
            "freshness": job.freshness.label.value,
            "freshness_evidence": _dump(job.freshness.evidence)
            if job.freshness.evidence
            else None,
            "body_text": job.body_text,
            "ats": job.ats.value if job.ats else None,
            "status": job.status.value,
            "signals": _dump(job.signals),
            "last_checked_at": job.last_checked_at.isoformat(),
        }
        res = (
            self._c.table("jobs")
            .upsert(row, on_conflict="dedup_hash")
            .execute()
        )
        return UUID(res.data[0]["id"])

    def get(self, job_id: UUID) -> Optional[ValidatedJob]:
        # Reading a full ValidatedJob from columns is lossy without raw_lead
        # provenance, so we keep this as a TODO until the orchestrator needs
        # round-trip reads. For now, return None to signal "not implemented".
        return None

    def find_by_dedup_hash(self, dedup_hash: str) -> Optional[UUID]:
        res = (
            self._c.table("jobs")
            .select("id")
            .eq("dedup_hash", dedup_hash)
            .limit(1)
            .execute()
        )
        if not res.data:
            return None
        return UUID(res.data[0]["id"])

    def mark_status(self, job_id: UUID, status: str) -> None:
        self._c.table("jobs").update({"status": status}).eq("id", str(job_id)).execute()


class SupabaseRunRepo:
    def __init__(self, client) -> None:
        self._c = client

    def insert(self, run: RunRecord) -> UUID:
        row = {
            "id": str(run.id),
            "profile_id": str(run.profile_id) if run.profile_id else None,
            "criteria": _dump(run.criteria),
            "plan": _dump(run.plan) if run.plan else None,
            "llm_provider": run.llm_provider,
            "llm_model": run.llm_model,
            "search_provider": run.search_provider,
            "latency_ms": run.metrics.latency_ms,
            "tokens_in": run.metrics.tokens_in,
            "tokens_out": run.metrics.tokens_out,
            "cost_usd": run.metrics.cost_usd,
            "discovered": run.metrics.discovered,
            "validated": run.metrics.validated,
            "surfaced": run.metrics.surfaced,
            "rejection_log": [_dump(r) for r in run.rejection_log],
        }
        self._c.table("runs").insert(row).execute()
        return run.id

    def get(self, run_id: UUID) -> Optional[RunRecord]:
        # Round-trip reads are an M9 concern; not needed for v1 flow.
        return None

    def attach_report(self, run_id: UUID, job_id: UUID, report: JobReport) -> None:
        row = {
            "run_id": str(run_id),
            "job_id": str(job_id),
            "match_score": report.score.match_score,
            "urgency_score": report.score.urgency_score,
            "match_rationale": report.score.match_rationale,
            "concerns": report.score.concerns,
            "application_angle": report.score.application_angle,
            "outreach_angle": report.score.outreach_angle,
            "evidence": [_dump(c) for c in report.evidence.citations],
            "red_team_decision": (
                "accept"
                if (report.red_team and report.red_team.accepted)
                else "reject"
            ),
            "red_team_reasons": report.red_team.reasons if report.red_team else [],
        }
        self._c.table("run_jobs").upsert(row, on_conflict="run_id,job_id").execute()


class SupabaseUserActionsRepo:
    def __init__(self, client) -> None:
        self._c = client

    def get(self, profile_id: UUID, job_id: UUID) -> Optional[UserActions]:
        res = (
            self._c.table("user_actions")
            .select("*")
            .eq("profile_id", str(profile_id))
            .eq("job_id", str(job_id))
            .limit(1)
            .execute()
        )
        if not res.data:
            return None
        d = res.data[0]
        return UserActions(
            profile_id=UUID(d["profile_id"]),
            job_id=UUID(d["job_id"]),
            favorited=d["favorited"],
            applied=d["applied"],
            applied_at=d.get("applied_at"),
            notes=d.get("notes"),
        )

    def upsert(self, actions: UserActions) -> None:
        row = {
            "profile_id": str(actions.profile_id),
            "job_id": str(actions.job_id),
            "favorited": actions.favorited,
            "applied": actions.applied,
            "applied_at": actions.applied_at.isoformat() if actions.applied_at else None,
            "notes": actions.notes,
        }
        self._c.table("user_actions").upsert(
            row, on_conflict="profile_id,job_id"
        ).execute()

    def list_for_profile(self, profile_id: UUID) -> list[UserActions]:
        res = (
            self._c.table("user_actions")
            .select("*")
            .eq("profile_id", str(profile_id))
            .execute()
        )
        out: list[UserActions] = []
        for d in res.data or []:
            out.append(
                UserActions(
                    profile_id=UUID(d["profile_id"]),
                    job_id=UUID(d["job_id"]),
                    favorited=d["favorited"],
                    applied=d["applied"],
                    applied_at=d.get("applied_at"),
                    notes=d.get("notes"),
                )
            )
        return out


class SupabaseRepositories:
    """Concrete `Repositories` backed by a single Supabase client."""

    def __init__(self, url: str, anon_key: str) -> None:
        self._client = _client(url, anon_key)
        self.profiles = SupabaseProfileRepo(self._client)
        self.jobs = SupabaseJobRepo(self._client)
        self.runs = SupabaseRunRepo(self._client)
        self.user_actions = SupabaseUserActionsRepo(self._client)
