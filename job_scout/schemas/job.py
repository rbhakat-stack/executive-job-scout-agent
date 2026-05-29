"""Validated job + evidence contracts.

A `ValidatedJob` is the only shape allowed to reach the Scoring Agent.
A `JobReport` (combining job + score + evidence) is the only shape the
Red Team Agent may admit to the user-facing surface.
"""
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from ._time import utc_now
from .common import ATS, FreshnessLabel, JobStatus, WorkMode
from .search import RawJobLead


# --- Validation primitives ---

class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    live: bool = Field(description="HTTP fetch succeeded AND posting not flagged closed.")
    http_status: Optional[int] = None
    final_url: Optional[HttpUrl] = Field(
        default=None,
        description="Post-redirect canonical URL. None if the fetch failed entirely.",
    )
    redirected: bool = False
    expired_signals: list[str] = Field(
        default_factory=list,
        description="Phrases like 'this role is no longer accepting' that triggered rejection.",
    )
    error: Optional[str] = None
    checked_at: datetime = Field(default_factory=utc_now)


class FreshnessEvidence(BaseModel):
    """How we know the posting date. Required when label != UNKNOWN."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        description=(
            "Where the date came from: 'ats_field', 'json_ld_datePosted', "
            "'meta_published_time', 'search_engine_date', 'page_text'."
        ),
    )
    snippet: Optional[str] = Field(
        default=None,
        description="The raw text/value that established the date.",
    )
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class Freshness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: FreshnessLabel
    posted_at: Optional[date] = None
    evidence: Optional[FreshnessEvidence] = None

    def model_post_init(self, _ctx) -> None:  # noqa: D401
        # Invariant: cannot label RECENT/OLDER without evidence.
        if self.label != FreshnessLabel.UNKNOWN and self.evidence is None:
            raise ValueError(
                "Freshness.label != UNKNOWN requires evidence. "
                "This is the 'no stale jobs as active' invariant."
            )


# --- Citation / Evidence ---

class Citation(BaseModel):
    """A direct quote tying a claim to source text."""

    model_config = ConfigDict(extra="forbid")

    claim_key: str = Field(
        description=(
            "Stable key identifying the claim, e.g. 'match.industry', "
            "'urgency.posted_recently', 'freshness.posted_at'."
        ),
    )
    quote: str = Field(min_length=1, description="Exact text from the source page.")
    source_url: HttpUrl
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citations: list[Citation] = Field(default_factory=list)

    def keys(self) -> set[str]:
        return {c.claim_key for c in self.citations}


# --- Signals captured during validation/parsing ---

class JobSignals(BaseModel):
    """Lightweight detected signals used by Scoring Agent."""

    model_config = ConfigDict(extra="forbid")

    recruiter_contact: Optional[str] = None
    multiple_openings: bool = False
    urgency_phrases: list[str] = Field(default_factory=list)
    transformation_phrases: list[str] = Field(default_factory=list)
    refreshed_posting: bool = False


# --- The fully validated job ---

class ValidatedJob(BaseModel):
    """A job posting that has passed HTTP + ATS validation.

    Only `ValidatedJob`s may proceed to Scoring. The dedup_hash MUST be set
    by the Validation Agent (see `src/validation/dedup.py`).
    """

    model_config = ConfigDict(extra="forbid")

    # Provenance
    raw_lead: RawJobLead
    validation: ValidationResult

    # Identity
    canonical_url: HttpUrl
    apply_url: HttpUrl
    dedup_hash: str = Field(min_length=16)

    # Core fields (all required after validation)
    title: str
    company: str
    body_text: str = Field(
        min_length=20,
        description="Extracted job description. Used by Scoring + Evidence agents.",
    )

    # Optional contextual fields
    location: Optional[str] = None
    work_mode: WorkMode = WorkMode.UNKNOWN
    ats: Optional[ATS] = None

    # Freshness
    freshness: Freshness

    # Detected signals
    signals: JobSignals = Field(default_factory=JobSignals)

    # Lifecycle
    status: JobStatus = JobStatus.ACTIVE
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_checked_at: datetime = Field(default_factory=utc_now)
