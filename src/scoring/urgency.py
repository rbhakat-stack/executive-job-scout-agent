"""Urgency score features.

Seven deterministic features, weights summing to 100. Same `ScoreFeature`
contract as match: non-zero contributions carry evidence keys.

Total weight = 25 + 15 + 15 + 10 + 10 + 10 + 15 = 100
"""
from __future__ import annotations

from datetime import date, timedelta

from src.schemas import ATS, ScoreFeature, ValidatedJob

_RELIABLE_ATS = (
    ATS.GREENHOUSE,
    ATS.LEVER,
    ATS.WORKDAY,
    ATS.ASHBY,
    ATS.SMARTRECRUITERS,
    ATS.ICIMS,
)


def compute_urgency_features(job: ValidatedJob, *, today: date) -> list[ScoreFeature]:
    features: list[ScoreFeature] = []

    # 1 + 2. Posted recency (25 + 15)
    posted = job.freshness.posted_at
    if posted is not None:
        age_days = (today - posted).days
        if age_days <= 7:
            features.append(
                ScoreFeature(
                    name="posted_within_7d",
                    weight=25,
                    contribution=25.0,
                    evidence_keys=["urgency.posted_within_7d", "freshness.posted_at"],
                    notes=f"posted {age_days} days ago",
                )
            )
            features.append(
                ScoreFeature(
                    name="posted_within_14d",
                    weight=15,
                    contribution=15.0,
                    evidence_keys=["urgency.posted_within_14d", "freshness.posted_at"],
                    notes="implied by posted_within_7d",
                )
            )
        elif age_days <= 14:
            features.append(
                ScoreFeature(
                    name="posted_within_7d",
                    weight=25,
                    contribution=0.0,
                    evidence_keys=[],
                    notes=f"posted {age_days} days ago",
                )
            )
            features.append(
                ScoreFeature(
                    name="posted_within_14d",
                    weight=15,
                    contribution=15.0,
                    evidence_keys=["urgency.posted_within_14d", "freshness.posted_at"],
                    notes=f"posted {age_days} days ago",
                )
            )
        else:
            features.append(
                ScoreFeature(
                    name="posted_within_7d",
                    weight=25,
                    contribution=0.0,
                    evidence_keys=[],
                    notes=f"posted {age_days} days ago",
                )
            )
            features.append(
                ScoreFeature(
                    name="posted_within_14d",
                    weight=15,
                    contribution=0.0,
                    evidence_keys=[],
                    notes=f"posted {age_days} days ago",
                )
            )
    else:
        # Unknown freshness means no points for recency — never penalize the
        # job into negative territory, but no points either.
        features.append(
            ScoreFeature(
                name="posted_within_7d",
                weight=25,
                contribution=0.0,
                evidence_keys=[],
                notes="posting date unknown",
            )
        )
        features.append(
            ScoreFeature(
                name="posted_within_14d",
                weight=15,
                contribution=0.0,
                evidence_keys=[],
                notes="posting date unknown",
            )
        )

    # 3. Urgency phrases (15)
    if job.signals.urgency_phrases:
        features.append(
            ScoreFeature(
                name="urgency_phrase_present",
                weight=15,
                contribution=15.0,
                evidence_keys=["urgency.phrase"],
                notes=f"matched: {job.signals.urgency_phrases}",
            )
        )
    else:
        features.append(
            ScoreFeature(
                name="urgency_phrase_present",
                weight=15,
                contribution=0.0,
                evidence_keys=[],
                notes="no urgency phrases on page",
            )
        )

    # 4. Recruiter contact (10)
    if job.signals.recruiter_contact:
        features.append(
            ScoreFeature(
                name="recruiter_contact_listed",
                weight=10,
                contribution=10.0,
                evidence_keys=["urgency.recruiter_contact"],
                notes=f"contact: {job.signals.recruiter_contact}",
            )
        )
    else:
        features.append(
            ScoreFeature(
                name="recruiter_contact_listed",
                weight=10,
                contribution=0.0,
                evidence_keys=[],
                notes="no recruiter contact in page",
            )
        )

    # 5. Multiple openings (10)
    if job.signals.multiple_openings:
        features.append(
            ScoreFeature(
                name="multiple_openings",
                weight=10,
                contribution=10.0,
                evidence_keys=["urgency.multiple_openings"],
                notes="'multiple openings' phrase detected",
            )
        )
    else:
        features.append(
            ScoreFeature(
                name="multiple_openings",
                weight=10,
                contribution=0.0,
                evidence_keys=[],
                notes="no multiple-openings phrase",
            )
        )

    # 6. Transformation phrases (10)
    if job.signals.transformation_phrases:
        features.append(
            ScoreFeature(
                name="transformation_phrase_present",
                weight=10,
                contribution=10.0,
                evidence_keys=["urgency.transformation_phrase"],
                notes=f"matched: {job.signals.transformation_phrases}",
            )
        )
    else:
        features.append(
            ScoreFeature(
                name="transformation_phrase_present",
                weight=10,
                contribution=0.0,
                evidence_keys=[],
                notes="no transformation phrases on page",
            )
        )

    # 7. ATS source reliability (15)
    if job.ats in _RELIABLE_ATS:
        features.append(
            ScoreFeature(
                name="ats_source_reliable",
                weight=15,
                contribution=15.0,
                evidence_keys=["urgency.source_ats"],
                notes=f"hosted on {job.ats.value}",
            )
        )
    else:
        # Non-ATS pages aren't worthless; they get partial credit.
        features.append(
            ScoreFeature(
                name="ats_source_reliable",
                weight=15,
                contribution=5.0,
                evidence_keys=[],
                notes="not on a known ATS host; partial credit",
            )
        )

    return features
