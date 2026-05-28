"""Validation Agent.

Turns a `RawJobLead` (which is just a search hit) into a `ValidatedJob`
(which has been HTTP-fetched, structurally validated, and freshness-tagged
with evidence). Only `ValidatedJob`s may proceed downstream.

If the lead cannot be validated the agent returns a `RejectionLogEntry`
instead, so the orchestrator can record why each surfaced-or-rejected lead
ended up where it did.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional

import httpx

from src.agents.search import detect_ats
from src.parsers.ats import extract_from_html
from src.schemas import (
    JobSignals,
    JobStatus,
    RawJobLead,
    RejectionLogEntry,
    SearchCriteria,
    ValidatedJob,
    ValidationResult,
    WorkMode,
)
from src.validation.dedup import canonicalize_url, compute_dedup_hash
from src.validation.freshness import infer_freshness
from src.validation.liveness import DEFAULT_USER_AGENT, fetch_url

# A job body shorter than this almost certainly didn't extract — likely a
# bot-blocked page, a login wall, or a redirect target.
_MIN_BODY_TEXT = 50


@dataclass
class ValidationOutcome:
    """Either a validated job or a rejection log entry. Exactly one is set."""

    job: Optional[ValidatedJob] = None
    rejection: Optional[RejectionLogEntry] = None

    @property
    def accepted(self) -> bool:
        return self.job is not None


def _reject(url: str, reason: str) -> ValidationOutcome:
    return ValidationOutcome(
        rejection=RejectionLogEntry(stage="validation", url=url, reason=reason)
    )


class ValidationAgent:
    """HTTP-validates a lead and emits a `ValidatedJob` or a rejection."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: int = 15,
        clock: Optional[Callable[[], date]] = None,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._timeout = timeout_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc).date())

    def validate(
        self,
        lead: RawJobLead,
        criteria: SearchCriteria,
    ) -> ValidationOutcome:
        url = str(lead.url)

        # 1. Fetch
        fetched = fetch_url(
            url,
            client=self._client,
            user_agent=self._user_agent,
            timeout_seconds=self._timeout,
        )
        if not fetched.ok:
            reason = (
                fetched.error
                if fetched.error
                else f"HTTP {fetched.status_code}"
            )
            return _reject(url, f"fetch failed: {reason}")

        final_url = fetched.final_url or url

        # 2. Extract structured content
        extracted = extract_from_html(fetched.body, source_url=final_url)

        # 3. Body length sanity (anything below this is a parser failure
        #    or a content gate, not a real posting).
        if len(extracted.body_text.strip()) < _MIN_BODY_TEXT:
            return _reject(final_url, "body text too short to be a real posting")

        # 4. Expired signals — checked BEFORE required-field rejection so
        #    an explicitly closed posting always rejects for "expired" and
        #    not for an incidental parser issue.
        if extracted.expired_signals:
            return _reject(
                final_url,
                f"expired posting: {extracted.expired_signals[0]!r}",
            )

        # 5. Required fields
        if not extracted.title or not extracted.title.strip():
            return _reject(final_url, "missing title")
        if not extracted.company or not extracted.company.strip():
            return _reject(final_url, "missing company")

        # 5. Freshness
        freshness = infer_freshness(
            extracted,
            search_engine_date=(
                lead.search_engine_date.date() if lead.search_engine_date else None
            ),
            today=self._clock(),
            max_age_days=criteria.max_age_days,
        )

        # 6. Dedup hash
        dedup_hash = compute_dedup_hash(
            canonicalize_url(final_url),
            extracted.title,
            extracted.company,
        )

        # 7. Signals
        signals = JobSignals(
            recruiter_contact=extracted.recruiter_contact,
            multiple_openings=extracted.multiple_openings,
            urgency_phrases=_detect_phrases(
                extracted.body_text,
                (
                    "urgently hiring",
                    "immediate start",
                    "immediate hire",
                    "actively hiring",
                ),
            ),
            transformation_phrases=_detect_phrases(
                extracted.body_text,
                (
                    "ai transformation",
                    "digital transformation",
                    "digital reinvention",
                    "data transformation",
                ),
            ),
        )

        # 8. Work mode
        work_mode = _coerce_work_mode(extracted.work_mode)

        # 9. ATS (detect from final URL — overrides lead.ats_guess)
        ats = detect_ats(final_url)

        # 10. Build the validated job
        try:
            job = ValidatedJob(
                raw_lead=lead,
                validation=ValidationResult(
                    live=True,
                    http_status=fetched.status_code,
                    final_url=final_url,
                    redirected=fetched.redirected,
                ),
                canonical_url=final_url,
                apply_url=extracted.apply_url or final_url,
                dedup_hash=dedup_hash,
                title=extracted.title.strip(),
                company=extracted.company.strip(),
                body_text=extracted.body_text.strip(),
                location=extracted.location,
                work_mode=work_mode,
                ats=ats,
                freshness=freshness,
                signals=signals,
                status=JobStatus.ACTIVE,
            )
        except Exception as e:  # pydantic validation error etc.
            return _reject(final_url, f"schema assembly failed: {e}")

        return ValidationOutcome(job=job)


def _detect_phrases(text: str, phrases: tuple[str, ...]) -> list[str]:
    if not text:
        return []
    t = text.lower()
    return [p for p in phrases if p in t]


def _coerce_work_mode(value: Optional[str]) -> WorkMode:
    if value == "remote":
        return WorkMode.REMOTE
    if value == "hybrid":
        return WorkMode.HYBRID
    if value == "onsite":
        return WorkMode.ONSITE
    return WorkMode.UNKNOWN
