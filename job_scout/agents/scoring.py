"""Scoring Agent.

Combines deterministic match + urgency features with an LLM-written
rationale. The LLM is optional: when absent or when its output fails the
'substantive rationale' check, the agent falls back to a deterministic
rationale built directly from the feature contributions.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable, Optional

from job_scout.llm import LLM
from job_scout.schemas import (
    CandidateProfile,
    ScoreFeature,
    ScoreResult,
    SearchCriteria,
    SeniorityLevel,
    ValidatedJob,
)
from job_scout.scoring.match import (
    _PROFILE_SENIORITY_RANK,
    _TITLE_SENIORITY_RANK,
    compute_match_features,
)
from job_scout.scoring.rationale import (
    deterministic_rationale,
    is_substantive_rationale,
    llm_rationale,
)
from job_scout.scoring.urgency import compute_urgency_features


# Seniority gap (|profile_rank - job_rank|) -> multiplier on the match
# score. A perfect rank match passes through unchanged; a wide gap
# (e.g. SVP candidate vs Manager-level role) is dampened sharply so
# junior roles can't bubble up purely on industry/tech overlap.
_SENIORITY_GAP_MULTIPLIER: dict[int, float] = {
    0: 1.00,
    1: 0.85,
    2: 0.50,
    3: 0.20,
    4: 0.10,
    5: 0.05,
}

# When the job title carries no recognized seniority phrase at all
# (e.g. "Account Executive", "Specialist"), apply a mild caution
# multiplier — we don't know the seniority, so don't fully trust the
# content overlap.
_SENIORITY_UNDETECTED_MULTIPLIER = 0.75


def _seniority_multiplier(
    *, job_title: str, profile_seniority: SeniorityLevel
) -> tuple[float, str]:
    """Return (multiplier, explanation) for the seniority-alignment correction."""
    jt = (job_title or "").lower()
    job_rank: Optional[int] = None
    matched_kw: Optional[str] = None
    for kw, rank in sorted(_TITLE_SENIORITY_RANK, key=lambda x: -len(x[0])):
        if kw in jt:
            job_rank = rank
            matched_kw = kw
            break

    if job_rank is None:
        return (
            _SENIORITY_UNDETECTED_MULTIPLIER,
            "no seniority phrase detected in title; mild caution applied",
        )

    profile_rank = _PROFILE_SENIORITY_RANK.get(profile_seniority, 0)
    delta = abs(profile_rank - job_rank)
    mult = _SENIORITY_GAP_MULTIPLIER.get(delta, 0.05)
    return (
        mult,
        (
            f"detected '{matched_kw}' (rank {job_rank}) vs profile "
            f"{profile_seniority.value} (rank {profile_rank}); "
            f"gap {delta} -> multiplier {mult:.2f}"
        ),
    )


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
        *,
        skip_llm: bool = False,
    ) -> ScoreResult:
        """Score a job (match + urgency) and produce a rationale.

        `skip_llm=True` forces the deterministic rationale path. Used by
        the orchestrator's two-pass flow so we don't burn LLM tokens (and
        hit rate limits) on jobs that will be filtered by the Red Team's
        match-score threshold anyway.
        """
        today = self._clock()
        match_features = compute_match_features(job, profile, criteria)
        urgency_features = compute_urgency_features(job, today=today)

        # Apply a seniority-alignment multiplier to the raw match score.
        # Without this, a junior role with strong industry/tech overlap
        # can score in the 40s-50s for an SVP candidate, which is wrong.
        raw_match = sum(f.contribution for f in match_features)
        seniority_mult, seniority_note = _seniority_multiplier(
            job_title=job.title, profile_seniority=profile.seniority_level,
        )
        match_score = max(0, min(100, round(raw_match * seniority_mult)))
        urgency_score = _clamp_score(urgency_features)

        # Try LLM first (if available and not skipped); fall back to
        # deterministic if the LLM is silent, malformed, rate-limited, or
        # produces a non-substantive rationale.
        rationale = concerns = app_angle = outreach = None
        if self._llm is not None and not skip_llm:
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
                seniority_multiplier=seniority_mult,
                seniority_note=seniority_note,
                raw_match_before_multiplier=int(round(raw_match)),
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
