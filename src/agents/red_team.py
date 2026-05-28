"""Red Team Agent.

The last gate before a report reaches the user. Applies the rejection rules
from `docs/ARCHITECTURE.md`:

  1. Apply URL missing
  2. Source URL not live (per `ValidationResult`)
  3. Posting older than `criteria.max_age_days` (unless `criteria.allow_older`)
  4. Job status is not ACTIVE
  5. Evidence gaps (any non-zero feature claim_key without a citation)
  6. Match rationale is generic (`is_substantive_rationale` check)
  7. Posting date claimed without evidence (defense in depth — the schema
     also blocks this structurally)
  8. Match score below `criteria.min_match_score`
  9. Duplicate of a job already accepted in this run

The agent is stateful only by design choice: it maintains `seen_dedup_hashes`
so rule #9 works across calls within a single run. The orchestrator creates
one agent per run and calls `evaluate()` per assembled report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.agents.report import ReportAssembly
from src.schemas import (
    FreshnessLabel,
    JobStatus,
    RedTeamDecision,
    SearchCriteria,
)
from src.scoring.rationale import is_substantive_rationale


# Stable rejection-reason strings. The Streamlit UI and observability log
# read these to drive filters / counters in M8 / M9.
class Reasons:
    APPLY_URL_MISSING = "apply URL missing"
    SOURCE_NOT_LIVE = "source URL not live"
    POSTING_TOO_OLD = "posting older than max_age_days"
    POSTING_DATE_UNCITED = "posting date claimed without evidence"
    JOB_NOT_ACTIVE = "job status is not active"
    EVIDENCE_GAP = "evidence gap"
    GENERIC_RATIONALE = "match rationale is generic"
    MATCH_BELOW_THRESHOLD = "match score below threshold"
    DUPLICATE_IN_RUN = "duplicate of a previously accepted job"


@dataclass
class RedTeamAgent:
    seen_dedup_hashes: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.seen_dedup_hashes.clear()

    def evaluate(
        self,
        assembly: ReportAssembly,
        criteria: SearchCriteria,
    ) -> RedTeamDecision:
        report = assembly.report
        job = report.job
        score = report.score

        reasons: list[str] = []

        # 1. Apply URL present
        if not job.apply_url:
            reasons.append(Reasons.APPLY_URL_MISSING)

        # 2. Source URL live (sanity check on Validation Agent's verdict)
        if not job.validation.live:
            reasons.append(Reasons.SOURCE_NOT_LIVE)

        # 3. Freshness threshold (only OLDER triggers rejection; UNKNOWN
        #    is left to the user — we don't know it's stale)
        if (
            job.freshness.label is FreshnessLabel.OLDER
            and not criteria.allow_older
        ):
            reasons.append(
                f"{Reasons.POSTING_TOO_OLD} ({criteria.max_age_days}d)"
            )

        # 4. Posting date fabricated (defense in depth)
        if (
            job.freshness.label
            in (FreshnessLabel.RECENT, FreshnessLabel.OLDER)
            and job.freshness.evidence is None
        ):
            reasons.append(Reasons.POSTING_DATE_UNCITED)

        # 5. Job still active
        if job.status is not JobStatus.ACTIVE:
            reasons.append(f"{Reasons.JOB_NOT_ACTIVE} ({job.status.value})")

        # 6. Evidence gaps (every non-zero feature key must be cited)
        if assembly.coverage_gaps:
            reasons.append(
                f"{Reasons.EVIDENCE_GAP}: " + ", ".join(assembly.coverage_gaps)
            )

        # 7. Generic rationale
        if not is_substantive_rationale(
            score.match_rationale,
            job=job,
            match_features=score.match_features,
        ):
            reasons.append(Reasons.GENERIC_RATIONALE)

        # 8. Relevance threshold
        if score.match_score < criteria.min_match_score:
            reasons.append(
                f"{Reasons.MATCH_BELOW_THRESHOLD} "
                f"({score.match_score} < {criteria.min_match_score})"
            )

        # 9. Duplicate within this run
        if job.dedup_hash in self.seen_dedup_hashes:
            reasons.append(Reasons.DUPLICATE_IN_RUN)

        decision = RedTeamDecision(accepted=not reasons, reasons=reasons)

        # Only record the hash if we accepted — rejected jobs shouldn't
        # block their own near-duplicates from being considered.
        if decision.accepted:
            self.seen_dedup_hashes.add(job.dedup_hash)

        return decision
