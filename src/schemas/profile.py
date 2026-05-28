"""CandidateProfile: the structured extraction from CV + LinkedIn.

Built by the Profile Agent (M2). Consumed by Planner, Scoring, and Evidence
agents. Persisted to `profiles` table.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ._time import utc_now
from .common import SeniorityLevel


class CandidateProfile(BaseModel):
    """Structured candidate profile derived from resume + LinkedIn."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # --- Identity / provenance ---
    resume_filename: Optional[str] = None
    resume_text_sha256: str = Field(
        description=(
            "Hash of the extracted resume TEXT (not the original bytes). "
            "Used for dedup and audit trail."
        ),
    )
    linkedin_url: Optional[str] = None
    extracted_at: datetime = Field(default_factory=utc_now)

    # --- Narrative ---
    summary: str = Field(description="Core experience summary, ~2-4 sentences.")

    # --- Domain coverage ---
    industries: list[str] = Field(default_factory=list)
    functional_expertise: list[str] = Field(default_factory=list)
    technical_expertise: list[str] = Field(default_factory=list)
    transformation_themes: list[str] = Field(default_factory=list)
    ai_data_cloud_experience: list[str] = Field(
        default_factory=list,
        description="AI, data, cloud, CRM, platform-level experience tags.",
    )

    # --- Scope / scale ---
    leadership_scope: Optional[str] = Field(
        default=None,
        description="Team size, geographic scope, etc.",
    )
    client_account_experience: list[str] = Field(default_factory=list)
    revenue_pl_scale: Optional[str] = Field(
        default=None,
        description="P&L or revenue scale signal if present in CV.",
    )

    # --- Seniority + targeting ---
    seniority_level: SeniorityLevel
    target_archetypes: list[str] = Field(
        default_factory=list,
        description=(
            "Higher-level job archetypes the profile fits, e.g. "
            "'Life Sciences Technology Partner', 'Chief Digital Officer'."
        ),
    )

    # --- Keyword bundles used by Planner and Scoring agents ---
    search_keywords: list[str] = Field(
        default_factory=list,
        description="Words/phrases the Search Agent should inject into queries.",
    )
    ranking_keywords: list[str] = Field(
        default_factory=list,
        description="Words/phrases the Scoring Agent uses to weight matches.",
    )
    title_equivalents: list[str] = Field(
        default_factory=list,
        description="Alternative titles that count as the same role.",
    )
