"""Match score features.

Nine deterministic features, weights summing to 100. Each feature returns a
`ScoreFeature` with its `weight`, `contribution` (<= weight), and the set of
`evidence_keys` the contribution is backed by. Non-zero contributions MUST
carry at least one evidence key — that is what makes the rationale citable
and the Red Team agent's 'generic rationale' rejection rule enforceable.

Total weight = 15 + 15 + 15 + 10 + 15 + 10 + 10 + 5 + 5 = 100
"""
from __future__ import annotations

import re
from typing import Iterable

from job_scout.schemas import (
    CandidateProfile,
    ScoreFeature,
    SearchCriteria,
    SeniorityLevel,
    ValidatedJob,
    WorkMode,
)

# Map seniority phrases that may appear in a job title to a numeric rank.
# Longer keys are checked first so 'senior director' beats 'director'.
_TITLE_SENIORITY_RANK: tuple[tuple[str, int], ...] = (
    ("executive vice president", 4),
    ("senior vice president", 4),
    ("vice president", 3),
    ("managing director", 3),
    ("senior director", 2),
    ("chief", 5),
    ("evp", 4),
    ("svp", 4),
    ("vp", 3),
    ("partner", 3),
    ("director", 2),
    ("head of", 2),
    ("principal", 2),
    ("senior manager", 1),
    ("manager", 1),
)

# Profile-level seniority -> numeric rank for the same scale.
_PROFILE_SENIORITY_RANK: dict[SeniorityLevel, int] = {
    SeniorityLevel.INDIVIDUAL_CONTRIBUTOR: 0,
    SeniorityLevel.MANAGER: 1,
    SeniorityLevel.SENIOR_MANAGER: 1,
    SeniorityLevel.DIRECTOR: 2,
    SeniorityLevel.SENIOR_DIRECTOR: 2,
    SeniorityLevel.VP: 3,
    SeniorityLevel.SVP: 4,
    SeniorityLevel.EVP: 4,
    SeniorityLevel.C_SUITE: 5,
    SeniorityLevel.PARTNER: 3,
    SeniorityLevel.MANAGING_DIRECTOR: 3,
}


def compute_match_features(
    job: ValidatedJob,
    profile: CandidateProfile,
    criteria: SearchCriteria,
) -> list[ScoreFeature]:
    """Compute the 9 match features for one (job, profile, criteria) triple."""
    haystack = ((job.title or "") + "\n" + (job.body_text or "")).lower()

    return [
        _overlap_feature(
            "industry_overlap",
            weight=15,
            needles=_concat(profile.industries, criteria.preferred_industries),
            haystack=haystack,
            evidence_key="match.industry",
        ),
        _overlap_feature(
            "functional_expertise_overlap",
            weight=15,
            needles=profile.functional_expertise,
            haystack=haystack,
            evidence_key="match.functional_expertise",
        ),
        _overlap_feature(
            "tech_domain_overlap",
            weight=15,
            needles=_concat(profile.technical_expertise, profile.ai_data_cloud_experience),
            haystack=haystack,
            evidence_key="match.tech_domain",
        ),
        _overlap_feature(
            "transformation_alignment",
            weight=10,
            needles=profile.transformation_themes,
            haystack=haystack,
            evidence_key="match.transformation",
        ),
        _title_match_feature(
            job_title=job.title,
            target_titles=_concat(
                criteria.target_titles,
                profile.target_archetypes,
                profile.title_equivalents,
            ),
        ),
        _seniority_feature(job_title=job.title, profile_seniority=profile.seniority_level),
        _location_feature(job=job, criteria=criteria),
        _must_have_keywords_feature(haystack, criteria.must_have_keywords),
        _exclusion_keywords_feature(haystack, criteria.exclusion_keywords),
    ]


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _overlap_feature(
    name: str,
    *,
    weight: int,
    needles: list[str],
    haystack: str,
    evidence_key: str,
) -> ScoreFeature:
    """Count how many needles appear as substrings in `haystack`. Saturate at 3."""
    if not needles:
        return ScoreFeature(
            name=name,
            weight=weight,
            contribution=0.0,
            evidence_keys=[],
            notes="no profile inputs to compare against",
        )
    matched = [n for n in needles if n.strip() and n.lower() in haystack]
    if not matched:
        return ScoreFeature(
            name=name,
            weight=weight,
            contribution=0.0,
            evidence_keys=[],
            notes=f"none of {len(needles)} expected terms found",
        )
    # Saturate at 3 matches so a single overlap still earns partial credit.
    contribution = min(float(weight), weight * len(matched) / 3.0)
    return ScoreFeature(
        name=name,
        weight=weight,
        contribution=round(contribution, 2),
        evidence_keys=[evidence_key],
        notes=f"matched {len(matched)} of {len(needles)} expected terms: {matched[:3]}",
    )


def _title_match_feature(*, job_title: str, target_titles: list[str]) -> ScoreFeature:
    weight = 15
    if not job_title or not target_titles:
        return ScoreFeature(
            name="title_match",
            weight=weight,
            contribution=0.0,
            evidence_keys=[],
            notes="no title or no target titles to compare",
        )
    jt = job_title.lower()
    # Exact substring match earns full credit.
    for t in target_titles:
        ts = (t or "").strip().lower()
        if ts and ts in jt:
            return ScoreFeature(
                name="title_match",
                weight=weight,
                contribution=float(weight),
                evidence_keys=["match.title"],
                notes=f"exact substring match: {t!r}",
            )
    # Token-set partial overlap.
    jt_tokens = _tokens(jt)
    best_overlap = 0
    best_t: str | None = None
    for t in target_titles:
        t_tokens = _tokens((t or "").lower())
        if not t_tokens:
            continue
        overlap = len(jt_tokens & t_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_t = t
    if best_overlap >= 2:
        contribution = round(weight * (best_overlap / (best_overlap + 1)), 2)
        return ScoreFeature(
            name="title_match",
            weight=weight,
            contribution=contribution,
            evidence_keys=["match.title"],
            notes=f"partial token overlap with {best_t!r}: {best_overlap} tokens",
        )
    return ScoreFeature(
        name="title_match",
        weight=weight,
        contribution=0.0,
        evidence_keys=[],
        notes=f"no overlap with target titles",
    )


def _seniority_feature(
    *, job_title: str, profile_seniority: SeniorityLevel
) -> ScoreFeature:
    weight = 10
    jt = (job_title or "").lower()
    job_rank: int | None = None
    matched_kw: str | None = None
    # Longest keys first so 'senior director' beats 'director'.
    for kw, rank in sorted(_TITLE_SENIORITY_RANK, key=lambda x: -len(x[0])):
        if kw in jt:
            job_rank = rank
            matched_kw = kw
            break
    if job_rank is None:
        return ScoreFeature(
            name="seniority_match",
            weight=weight,
            contribution=0.0,
            evidence_keys=[],
            notes="could not detect seniority in job title",
        )
    profile_rank = _PROFILE_SENIORITY_RANK.get(profile_seniority, 0)
    delta = abs(profile_rank - job_rank)
    if delta == 0:
        contribution = float(weight)
    elif delta == 1:
        contribution = float(weight) * 0.7
    elif delta == 2:
        contribution = float(weight) * 0.3
    else:
        contribution = 0.0
    return ScoreFeature(
        name="seniority_match",
        weight=weight,
        contribution=round(contribution, 2),
        evidence_keys=["match.seniority"] if contribution > 0 else [],
        notes=(
            f"detected '{matched_kw}' (rank {job_rank}) vs profile "
            f"{profile_seniority.value} (rank {profile_rank}); delta {delta}"
        ),
    )


def _location_feature(*, job: ValidatedJob, criteria: SearchCriteria) -> ScoreFeature:
    """Work-mode + location fit.

    Evidence-key handling: we ONLY claim `match.location_remote` when at
    least one subscore came from a positive match against criteria. Credit
    earned from "no constraint configured" or "unknown - can't penalize"
    is default-pass and carries no citation, because there's nothing
    concrete to cite. The Red Team's coverage-gap rule would otherwise
    flag these reports.
    """
    weight = 10
    parts: list[str] = []
    has_positive_evidence = False

    # Work-mode subscore (6 points).
    if criteria.work_modes:
        if job.work_mode in criteria.work_modes:
            work_mode_score = 6.0
            has_positive_evidence = True
            parts.append(f"work_mode '{job.work_mode.value}' matches criteria")
        elif job.work_mode is WorkMode.UNKNOWN:
            # Can't penalize what we don't know; no evidence either.
            work_mode_score = 3.0
            parts.append("work_mode unknown; partial credit (no penalty)")
        else:
            work_mode_score = 0.0
            parts.append(f"work_mode '{job.work_mode.value}' not in criteria")
    else:
        # User specified no work-mode preference; default-pass, no evidence.
        work_mode_score = 6.0
        parts.append("no work_mode criteria configured; full credit")

    # Location-preference subscore (4 points).
    if criteria.location_preference:
        if (
            job.location
            and criteria.location_preference.lower() in job.location.lower()
        ):
            loc_score = 4.0
            has_positive_evidence = True
            parts.append(
                f"location matches preference '{criteria.location_preference}'"
            )
        elif not job.location:
            loc_score = 2.0
            parts.append("job location unknown; partial credit (no penalty)")
        else:
            loc_score = 0.0
            parts.append(
                f"job location {job.location!r} does not contain "
                f"'{criteria.location_preference}'"
            )
    else:
        # No location preference set; default-pass, no evidence.
        loc_score = 4.0
        parts.append("no location preference configured; full credit")

    total = work_mode_score + loc_score
    return ScoreFeature(
        name="location_remote_fit",
        weight=weight,
        contribution=round(total, 2),
        evidence_keys=["match.location_remote"] if has_positive_evidence else [],
        notes="; ".join(parts),
    )


def _must_have_keywords_feature(haystack: str, keywords: list[str]) -> ScoreFeature:
    weight = 5
    if not keywords:
        return ScoreFeature(
            name="keyword_must_have",
            weight=weight,
            contribution=float(weight),
            evidence_keys=[],
            notes="no must-have keywords configured",
        )
    kws = [k.strip() for k in keywords if k.strip()]
    present = [k for k in kws if k.lower() in haystack]
    missing = [k for k in kws if k.lower() not in haystack]
    if not missing:
        return ScoreFeature(
            name="keyword_must_have",
            weight=weight,
            contribution=float(weight),
            evidence_keys=["match.keyword_must_have"],
            notes=f"all {len(kws)} must-have keywords present",
        )
    if not present:
        return ScoreFeature(
            name="keyword_must_have",
            weight=weight,
            contribution=0.0,
            evidence_keys=[],
            notes=f"no must-have keywords present (expected {kws})",
        )
    contribution = round(weight * len(present) / len(kws), 2)
    return ScoreFeature(
        name="keyword_must_have",
        weight=weight,
        contribution=contribution,
        evidence_keys=["match.keyword_must_have"],
        notes=f"{len(present)}/{len(kws)} must-haves present; missing {missing}",
    )


def _exclusion_keywords_feature(haystack: str, keywords: list[str]) -> ScoreFeature:
    weight = 5
    if not keywords:
        return ScoreFeature(
            name="keyword_exclusion",
            weight=weight,
            contribution=float(weight),
            evidence_keys=[],
            notes="no exclusion keywords configured",
        )
    hits = [k for k in keywords if k.strip() and k.lower() in haystack]
    if not hits:
        return ScoreFeature(
            name="keyword_exclusion",
            weight=weight,
            contribution=float(weight),
            evidence_keys=["match.keyword_exclusion"],
            notes="no exclusion keywords found",
        )
    return ScoreFeature(
        name="keyword_exclusion",
        weight=weight,
        contribution=0.0,
        evidence_keys=[],
        notes=f"contains excluded keywords: {hits}",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", s.lower()))


def _concat(*lists: Iterable[str]) -> list[str]:
    out: list[str] = []
    for lst in lists:
        if lst:
            out.extend(lst)
    return out
