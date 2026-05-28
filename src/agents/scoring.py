"""Scoring Agent.

Combines deterministic match + urgency features with an LLM-written
rationale. The LLM is optional: when absent or when its output fails the
'substantive rationale' check, the agent falls back to a deterministic
rationale built directly from the feature contributions.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable, Optional

from src.llm import LLM
from src.schemas import (
    CandidateProfile,
    ScoreFeature,
    ScoreResult,
    SearchCriteria,
    ValidatedJob,
)
from src.scoring.match import compute_match_features
from src.scoring.rationale import (
    deterministic_rationale,
    is_substantive_rationale,
    llm_rationale,
)
from src.scoring.urgency import compute_urgency_features


class ScoringAgent:
    def __init__(
        self,
        llm: Optional[LLM] = None,
        *,
        clock: Optional[Callable[[], date]] = None,
    ) -> None:
        self._llm = llm
        self._clock = clock or (lambda: datetime.now(timezone.utc).date())

    def score(
        self,
        job: ValidatedJob,
        profile: CandidateProfile,
        criteria: SearchCriteria,
    ) -> ScoreResult:
        today = self._clock()
        match_features = compute_match_features(job, profile, criteria)
        urgency_features = compute_urgency_features(job, today=today)
        match_score = _clamp_score(match_features)
        urgency_score = _clamp_score(urgency_features)

        # Try LLM first (if available); fall back to deterministic if the
        # LLM is silent, malformed, or produces a non-substantive rationale.
        rationale = concerns = app_angle = outreach = None
        if self._llm is not None:
            rationale, concerns, app_angle, outreach = llm_rationale(
                self._llm,
                job=job,
                profile=profile,
                criteria=criteria,
                match_features=match_features,
                urgency_features=urgency_features,
                match_score=match_score,
                urgency_score=urgency_score,
            )
            if not is_substantive_rationale(
                rationale, job=job, match_features=match_features
            ):
                rationale = concerns = app_angle = outreach = None

        if rationale is None:
            rationale, concerns, app_angle, outreach = deterministic_rationale(
                job=job,
                profile=profile,
                match_features=match_features,
                urgency_features=urgency_features,
                match_score=match_score,
                urgency_score=urgency_score,
            )

        return ScoreResult(
            match_score=match_score,
            urgency_score=urgency_score,
            match_features=match_features,
            urgency_features=urgency_features,
            match_rationale=rationale,
            concerns=concerns,
            application_angle=app_angle,
            outreach_angle=outreach,
        )


def _clamp_score(features: list[ScoreFeature]) -> int:
    total = sum(f.contribution for f in features)
    return max(0, min(100, round(total)))
