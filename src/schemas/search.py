"""Search Agent IO contracts.

`SearchPlan` is produced by the Planner Agent. `RawJobLead` is what the
Search Agent emits before validation/scoring.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from ._time import utc_now
from .common import ATS, SearchProviderName, SearchStrategy


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: SearchStrategy
    text: str = Field(description="The actual query string sent to the search provider.")
    site_filter: Optional[str] = Field(
        default=None,
        description="Optional site: filter, e.g. 'greenhouse.io'.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Why this query was chosen. For debug/UI display.",
    )
    expected_recency_days: int = 14


class SearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[SearchQuery]
    planner_notes: Optional[str] = None


class RawJobLead(BaseModel):
    """A raw lead from the search provider, BEFORE HTTP validation.

    A `RawJobLead` MUST NOT be surfaced to the user. It is the input to the
    Validation Agent. Only `ValidatedJob` instances may pass downstream.
    """

    model_config = ConfigDict(extra="forbid")

    title_guess: str = Field(description="Title as reported by the search result; unverified.")
    url: HttpUrl
    snippet: Optional[str] = None
    source_provider: SearchProviderName
    source_query: SearchQuery
    discovered_at: datetime = Field(default_factory=utc_now)
    ats_guess: Optional[ATS] = Field(
        default=None,
        description="Detected from URL pattern; may be wrong - Validation Agent confirms.",
    )
    search_engine_date: Optional[datetime] = Field(
        default=None,
        description="If the search provider returned a result date, capture it here.",
    )
