"""Final report shapes consumed by the UI + persisted by the DB layer."""
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from ._time import utc_now
from .criteria import SearchCriteria
from .job import EvidenceBundle, ValidatedJob
from .scoring import ScoreResult
from .search import SearchPlan


class RedTeamDecision(BaseModel):
    """Outcome of the Red Team Agent for a single JobReport."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    reasons: list[str] = Field(
        default_factory=list,
        description="Rejection reasons (free-form, stable strings).",
    )


class JobReport(BaseModel):
    """The fully assembled, citation-checked output for ONE job."""

    model_config = ConfigDict(extra="forbid")

    job: ValidatedJob
    score: ScoreResult
    evidence: EvidenceBundle
    red_team: Optional[RedTeamDecision] = None


class RejectionLogEntry(BaseModel):
    """One rejection in the run log (any stage)."""

    model_config = ConfigDict(extra="forbid")

    stage: str  # 'search', 'validation', 'scoring', 'red_team'
    url: Optional[str] = None
    reason: str


class RunMetrics(BaseModel):
    """Telemetry for one search run."""

    model_config = ConfigDict(extra="forbid")

    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    discovered: int = 0
    validated: int = 0
    surfaced: int = 0


class RunRecord(BaseModel):
    """One search run end-to-end."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)
    profile_id: Optional[UUID] = None
    criteria: SearchCriteria
    plan: Optional[SearchPlan] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    search_provider: Optional[str] = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    rejection_log: list[RejectionLogEntry] = Field(default_factory=list)
    reports: list[JobReport] = Field(default_factory=list)


class UserActions(BaseModel):
    """Per-(profile, job) user state stored in `user_actions`."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    profile_id: UUID
    favorited: bool = False
    applied: bool = False
    applied_at: Optional[datetime] = None
    notes: Optional[str] = None
    updated_at: datetime = Field(default_factory=utc_now)
