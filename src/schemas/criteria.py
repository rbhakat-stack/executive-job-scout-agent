"""SearchCriteria: the user-supplied targeting + thresholds for a run.

Built from the UI form. Threaded into Planner, Search, Validation, Scoring,
and Red Team agents.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import SeniorityLevel, WorkMode


class SearchCriteria(BaseModel):
    """Everything the user can tune for a search run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # --- Targeting ---
    target_titles: list[str] = Field(default_factory=list)
    preferred_industries: list[str] = Field(default_factory=list)
    preferred_companies: list[str] = Field(default_factory=list)
    excluded_companies: list[str] = Field(default_factory=list)
    seniority_levels: list[SeniorityLevel] = Field(default_factory=list)

    # --- Geography / mode ---
    location_preference: Optional[str] = None
    work_modes: list[WorkMode] = Field(
        default_factory=lambda: [WorkMode.REMOTE, WorkMode.HYBRID, WorkMode.ONSITE]
    )
    travel_tolerance: Optional[str] = Field(
        default=None,
        description="Free-form description, e.g. '<25% domestic, no intl.'.",
    )

    # --- Compensation ---
    comp_min_usd: Optional[int] = None
    comp_max_usd: Optional[int] = None

    # --- Keyword filters ---
    must_have_keywords: list[str] = Field(default_factory=list)
    exclusion_keywords: list[str] = Field(default_factory=list)

    # --- Thresholds enforced by Validation + Red Team ---
    max_age_days: int = Field(
        default=14,
        ge=1,
        le=365,
        description="Posting freshness window. Default 14d.",
    )
    allow_older: bool = Field(
        default=False,
        description="If false, the Red Team agent rejects postings older than max_age_days.",
    )
    prioritize_urgent: bool = Field(
        default=True,
        description="Boost urgency-flagged postings in the final ranking.",
    )
    min_match_score: int = Field(
        default=60,
        ge=0,
        le=100,
        description="Red Team rejects postings below this match score.",
    )

    @field_validator("comp_max_usd")
    @classmethod
    def _validate_comp_range(cls, v: Optional[int], info) -> Optional[int]:
        comp_min = info.data.get("comp_min_usd")
        if v is not None and comp_min is not None and v < comp_min:
            raise ValueError("comp_max_usd must be >= comp_min_usd")
        return v
