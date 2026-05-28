"""Scoring Agent IO contracts."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ScoreFeature(BaseModel):
    """One contributing feature to a match/urgency score."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="e.g. 'industry_overlap', 'posted_within_7d'.")
    weight: float = Field(
        ge=0.0,
        description="Maximum points this feature can contribute (pre-normalization).",
    )
    contribution: float = Field(
        ge=0.0,
        description="Actual points earned, <= weight.",
    )
    evidence_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Citation keys this feature relies on. The Scoring Agent rejects "
            "any non-zero contribution that has no backing citation."
        ),
    )
    notes: Optional[str] = None


class ScoreResult(BaseModel):
    """Match + urgency scores plus the user-facing rationale."""

    model_config = ConfigDict(extra="forbid")

    match_score: int = Field(ge=0, le=100)
    urgency_score: int = Field(ge=0, le=100)

    match_features: list[ScoreFeature] = Field(default_factory=list)
    urgency_features: list[ScoreFeature] = Field(default_factory=list)

    match_rationale: str = Field(
        min_length=20,
        description=(
            "Plain-English explanation of the match. MUST reference specific "
            "evidence keys; Red Team Agent rejects generic rationales."
        ),
    )
    concerns: Optional[str] = Field(
        default=None,
        description="Potential gaps or risks — also evidence-backed when possible.",
    )
    application_angle: Optional[str] = None
    outreach_angle: Optional[str] = None
