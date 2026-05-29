"""Rationale generation for the Scoring Agent.

Two paths:
  * `deterministic_rationale()`: synthesizes a plain-English explanation
    directly from the feature contributions. Never generic — it names
    specific features and their notes. Used as the floor.
  * `llm_rationale()`: asks an LLM to write a more readable explanation
    given the same feature data. The agent falls back to deterministic
    if the LLM is silent, parse-fails, or produces non-substantive text.
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

from src.llm import LLM
from src.schemas import (
    CandidateProfile,
    ScoreFeature,
    SearchCriteria,
    ValidatedJob,
)

RationaleTuple = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]


# ---------------------------------------------------------------------------
# Deterministic floor
# ---------------------------------------------------------------------------

def deterministic_rationale(
    *,
    job: ValidatedJob,
    profile: CandidateProfile,
    match_features: list[ScoreFeature],
    urgency_features: list[ScoreFeature],
    match_score: int,
    urgency_score: int,
    seniority_multiplier: float = 1.0,
    seniority_note: Optional[str] = None,
    raw_match_before_multiplier: Optional[int] = None,
) -> RationaleTuple:
    strong = [f for f in match_features if f.weight > 0 and f.contribution >= f.weight * 0.6]
    weak = [
        f
        for f in match_features
        if f.weight >= 5 and f.contribution < f.weight * 0.3 and f.name != "keyword_exclusion"
    ]

    parts = [
        f"Match score {match_score}/100 for the {job.title} role at {job.company}."
    ]

    # Surface the seniority adjustment so users see WHY a strong-on-paper
    # match was dampened (or boosted by being the right level).
    if (
        seniority_multiplier < 1.0
        and raw_match_before_multiplier is not None
        and raw_match_before_multiplier > match_score
    ):
        parts.append(
            f"Score reduced from {raw_match_before_multiplier}/100 to "
            f"{match_score}/100 by a x{seniority_multiplier:.2f} seniority "
            f"alignment multiplier ({seniority_note})."
        )
    for f in strong:
        parts.append(
            f"Strong on {_humanize(f.name)} "
            f"({f.contribution:.0f}/{f.weight}): {f.notes}."
        )
    if not strong:
        parts.append(
            "No feature crossed the 60% threshold for a strong signal; "
            "see the per-feature breakdown for nuance."
        )
    if urgency_score >= 60:
        parts.append(
            f"Urgency {urgency_score}/100 indicates active hiring pressure."
        )
    elif urgency_score < 30:
        parts.append(
            f"Urgency {urgency_score}/100 - this role does not appear to be "
            f"actively pushing for an immediate hire."
        )

    rationale = " ".join(parts)

    concerns: Optional[str] = None
    if weak:
        concerns = "Gaps: " + ", ".join(
            f"{_humanize(f.name)} ({f.contribution:.0f}/{f.weight})" for f in weak
        )

    primary_target = (
        profile.target_archetypes[0]
        if profile.target_archetypes
        else "the candidate's target archetype"
    )
    application_angle = (
        f"Lead with the experience that maps most directly to {primary_target}; "
        "in the cover or first message, name the specific evidence the Match "
        "Score is built on."
    )
    if strong:
        outreach_angle = (
            f"In outreach, reference the {_humanize(strong[0].name)} alignment "
            f"({strong[0].notes})."
        )
    else:
        outreach_angle = (
            "In outreach, lead with broad transformation themes from your CV; "
            "the specific overlap with this posting is thin."
        )

    return rationale, concerns, application_angle, outreach_angle


def _humanize(feature_name: str) -> str:
    return feature_name.replace("_", " ")


# ---------------------------------------------------------------------------
# LLM-assisted rationale
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = (
    "You are the Scoring Agent in an executive job-search system. "
    "Given a job posting, a candidate profile, and the deterministic "
    "feature contributions that have ALREADY been computed for you, "
    "write a single JSON object with exactly these keys: "
    "rationale, concerns, application_angle, outreach_angle. "
    "Rules: "
    "1. The `rationale` MUST mention the job's company and title and "
    "reference at least one of the feature names or evidence_keys you were given. "
    "2. Do NOT invent facts not present in the feature notes or job text. "
    "3. Do NOT be generic - statements like 'strong fit' or 'good match' "
    "without specific feature references will be rejected. "
    "4. Output ONLY the JSON object. No prose. No markdown fences."
)


def llm_rationale(
    llm: LLM,
    *,
    job: ValidatedJob,
    profile: CandidateProfile,
    criteria: SearchCriteria,
    match_features: list[ScoreFeature],
    urgency_features: list[ScoreFeature],
    match_score: int,
    urgency_score: int,
) -> RationaleTuple:
    """Ask the LLM for a rationale. Returns (None, None, None, None) on any failure."""
    nonzero = [f for f in match_features + urgency_features if f.contribution > 0]
    feature_summary = "\n".join(
        f"  - {f.name}: contribution={f.contribution:.1f}/{f.weight}, "
        f"evidence_keys={f.evidence_keys}, notes={f.notes!r}"
        for f in nonzero
    )
    user = (
        f"Job: {job.title} at {job.company} ({job.location or 'location unknown'}).\n"
        f"Work mode: {job.work_mode.value}.\n"
        f"Candidate seniority: {profile.seniority_level.value}.\n"
        f"Candidate industries: {profile.industries}\n"
        f"Candidate target archetypes: {profile.target_archetypes}\n"
        f"Match score: {match_score}/100. Urgency score: {urgency_score}/100.\n\n"
        "Nonzero feature contributions (you may only reference these as evidence):\n"
        f"{feature_summary or '  (none above zero)'}\n\n"
        "Job body excerpt (truncated to 1500 chars):\n"
        f"{(job.body_text or '')[:1500]}\n"
    )

    try:
        resp = llm.complete(system=LLM_SYSTEM_PROMPT, user=user)
        obj = json.loads(resp.text)
    except Exception:
        return None, None, None, None

    if not isinstance(obj, dict):
        return None, None, None, None
    return (
        obj.get("rationale") or None,
        obj.get("concerns") or None,
        obj.get("application_angle") or None,
        obj.get("outreach_angle") or None,
    )


# ---------------------------------------------------------------------------
# Substantive-rationale heuristic (Red Team uses the same check)
# ---------------------------------------------------------------------------

def is_substantive_rationale(
    rationale: Optional[str],
    *,
    job: ValidatedJob,
    match_features: list[ScoreFeature],
) -> bool:
    """A 'substantive' rationale meets all of:
      - >= 30 characters of content
      - mentions the job's company or a meaningful token from the job title
      - references at least one feature name or evidence key that earned
        a non-zero contribution
    """
    if not rationale or len(rationale.strip()) < 30:
        return False
    rl = rationale.lower()

    company_token = (job.company or "").split()
    company_present = bool(company_token) and company_token[0].lower() in rl
    title_token_present = any(
        tok.lower() in rl for tok in (job.title or "").split() if len(tok) > 3
    )
    if not (company_present or title_token_present):
        return False

    nonzero = [f for f in match_features if f.contribution > 0]
    if not nonzero:
        # If literally nothing scored, a rationale CAN'T cite a feature.
        # In that case length + job mention is enough — the deterministic
        # path will say so.
        return True
    feature_mentioned = any(
        f.name.replace("_", " ") in rl
        or f.name in rl
        or any(key in rl for key in f.evidence_keys)
        for f in nonzero
    )
    return feature_mentioned
