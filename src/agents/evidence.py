"""Evidence Agent.

Given a `ValidatedJob`, a `CandidateProfile`, a `SearchCriteria`, and the
`ScoreResult` produced by the Scoring Agent, the Evidence Agent walks the
`evidence_keys` referenced by every non-zero feature and produces a
`Citation` for each — backed by a real text span from the job page or a
URL/date pulled directly from validated metadata.

Invariant the Red Team Agent (M7) enforces: every `claim_key` in any
feature's `evidence_keys` MUST appear in `bundle.keys()`. `coverage_gaps()`
exposes any keys we couldn't back so the Red Team can act on them.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from src.schemas import (
    ATS,
    CandidateProfile,
    Citation,
    EvidenceBundle,
    ScoreResult,
    SearchCriteria,
    ValidatedJob,
)

# Window of context around a needle when extracting a snippet. Keep small
# enough to be quote-like, large enough to be meaningful in the UI.
_SNIPPET_WINDOW = 60

# Title-detected seniority phrases (mirror of src/scoring/match._TITLE_SENIORITY_RANK).
_SENIORITY_PHRASES: tuple[str, ...] = (
    "executive vice president",
    "senior vice president",
    "vice president",
    "managing director",
    "senior director",
    "chief",
    "evp",
    "svp",
    "vp",
    "partner",
    "director",
    "head of",
    "principal",
    "senior manager",
    "manager",
)

# Work-mode phrases the ATS parser uses for inference. Mirrored so the
# Evidence Agent can produce a citable span when work_mode was inferred
# from body text rather than from a location string.
_WORK_MODE_PHRASES: tuple[str, ...] = (
    "fully remote",
    "100% remote",
    "remote (",
    "remote -",
    "remote,",
    "work from anywhere",
    "hybrid",
    "on-site",
    "onsite",
    "in office",
)

# 'Multiple openings' patterns (same regex as the ATS parser used).
_MULTIPLE_OPENINGS_PATTERN = re.compile(
    r"\b(multiple|several|many)\s+(openings|positions|roles)\b", re.IGNORECASE
)


class EvidenceAgent:
    def extract(
        self,
        *,
        job: ValidatedJob,
        profile: CandidateProfile,
        criteria: SearchCriteria,
        score: ScoreResult,
    ) -> EvidenceBundle:
        # Collect every key the score claims to be backed by.
        keys_needed: list[str] = []
        for f in score.match_features + score.urgency_features:
            if f.contribution > 0:
                for k in f.evidence_keys:
                    if k not in keys_needed:
                        keys_needed.append(k)

        citations: list[Citation] = []
        for key in keys_needed:
            citations.extend(self._for_key(key, job=job, profile=profile, criteria=criteria))

        return EvidenceBundle(citations=citations)

    # ---------------------------------------------------------------------

    def _for_key(
        self,
        key: str,
        *,
        job: ValidatedJob,
        profile: CandidateProfile,
        criteria: SearchCriteria,
    ) -> list[Citation]:
        body = job.body_text or ""
        title = job.title or ""
        haystack = title + "\n" + body
        url = str(job.canonical_url)

        # --- match.* ---
        if key == "match.industry":
            return _quote_first(
                needles=_concat(profile.industries, criteria.preferred_industries),
                haystack=haystack,
                claim_key=key,
                source_url=url,
            )
        if key == "match.functional_expertise":
            return _quote_first(
                needles=profile.functional_expertise,
                haystack=haystack,
                claim_key=key,
                source_url=url,
            )
        if key == "match.tech_domain":
            return _quote_first(
                needles=_concat(profile.technical_expertise, profile.ai_data_cloud_experience),
                haystack=haystack,
                claim_key=key,
                source_url=url,
            )
        if key == "match.transformation":
            return _quote_first(
                needles=profile.transformation_themes,
                haystack=haystack,
                claim_key=key,
                source_url=url,
            )
        if key == "match.title":
            return [Citation(claim_key=key, quote=title, source_url=url)] if title else []
        if key == "match.seniority":
            phrase = _find_seniority_phrase(title)
            return (
                [Citation(claim_key=key, quote=phrase, source_url=url)]
                if phrase
                else []
            )
        if key == "match.location_remote":
            out: list[Citation] = []
            if job.location:
                out.append(
                    Citation(claim_key=key, quote=job.location, source_url=url)
                )
            # Also try to quote a work-mode phrase from the body text.
            wm = _quote_first(
                needles=list(_WORK_MODE_PHRASES),
                haystack=body,
                claim_key=key,
                source_url=url,
            )
            out.extend(wm)
            return out
        if key == "match.keyword_must_have":
            return _quote_first(
                needles=criteria.must_have_keywords,
                haystack=haystack,
                claim_key=key,
                source_url=url,
            )

        # --- urgency.* / freshness.* ---
        if key in ("urgency.posted_within_7d", "urgency.posted_within_14d", "freshness.posted_at"):
            ev = job.freshness.evidence
            quote = (
                ev.snippet
                if ev and ev.snippet
                else (job.freshness.posted_at.isoformat() if job.freshness.posted_at else None)
            )
            return [Citation(claim_key=key, quote=quote, source_url=url)] if quote else []
        if key == "urgency.phrase":
            return _quote_first(
                needles=list(job.signals.urgency_phrases),
                haystack=body,
                claim_key=key,
                source_url=url,
            )
        if key == "urgency.recruiter_contact":
            if job.signals.recruiter_contact:
                return [
                    Citation(
                        claim_key=key,
                        quote=job.signals.recruiter_contact,
                        source_url=url,
                    )
                ]
            return []
        if key == "urgency.multiple_openings":
            m = _MULTIPLE_OPENINGS_PATTERN.search(body)
            if m:
                start = max(0, m.start() - _SNIPPET_WINDOW)
                end = min(len(body), m.end() + _SNIPPET_WINDOW)
                return [
                    Citation(
                        claim_key=key,
                        quote=body[start:end].strip(),
                        source_url=url,
                        start_idx=m.start(),
                        end_idx=m.end(),
                    )
                ]
            return []
        if key == "urgency.transformation_phrase":
            return _quote_first(
                needles=list(job.signals.transformation_phrases),
                haystack=body,
                claim_key=key,
                source_url=url,
            )
        if key == "urgency.source_ats":
            if job.ats and job.ats is not ATS.OTHER:
                # The URL itself IS the evidence: the host pattern is what
                # proved this is a real ATS posting.
                return [Citation(claim_key=key, quote=url, source_url=url)]
            return []

        # Unknown key — caller (Red Team) sees the gap.
        return []


def coverage_gaps(score: ScoreResult, bundle: EvidenceBundle) -> list[str]:
    """Return claim_keys that earned a non-zero contribution but have no citation."""
    covered = bundle.keys()
    gaps: list[str] = []
    for f in score.match_features + score.urgency_features:
        if f.contribution <= 0:
            continue
        for k in f.evidence_keys:
            if k not in covered and k not in gaps:
                gaps.append(k)
    return gaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quote_first(
    *,
    needles: list[str],
    haystack: str,
    claim_key: str,
    source_url: str,
) -> list[Citation]:
    """Find the first needle that occurs in haystack; quote a context window."""
    if not haystack:
        return []
    lh = haystack.lower()
    for n in needles or []:
        if not n or not n.strip():
            continue
        idx = lh.find(n.lower())
        if idx == -1:
            continue
        start = max(0, idx - _SNIPPET_WINDOW)
        end = min(len(haystack), idx + len(n) + _SNIPPET_WINDOW)
        quote = haystack[start:end].strip()
        if not quote:
            continue
        return [
            Citation(
                claim_key=claim_key,
                quote=quote,
                source_url=source_url,
                start_idx=idx,
                end_idx=idx + len(n),
            )
        ]
    return []


def _find_seniority_phrase(title: str) -> Optional[str]:
    if not title:
        return None
    lt = title.lower()
    # Longest match first.
    for phrase in sorted(_SENIORITY_PHRASES, key=len, reverse=True):
        idx = lt.find(phrase)
        if idx != -1:
            return title[idx : idx + len(phrase)]
    return None


def _concat(*lists: Iterable[str]) -> list[str]:
    out: list[str] = []
    for lst in lists:
        if lst:
            out.extend(lst)
    return out
