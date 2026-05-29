"""Pipeline orchestrator.

Threads Planner -> Search -> Validation -> Scoring -> Evidence -> Report -> RedTeam
in a single deterministic loop. The orchestrator is the only place that
knows about all the agents together; the UI (M8) and any future API surface
talk to it via `Orchestrator.run(profile, criteria)`.

Design rules:
  * Pure dependency injection. LLM, search provider, HTTP client, and clock
    are all parameters. The orchestrator owns no global state.
  * Errors are recorded in `RunRecord.rejection_log` per (lead, stage) - the
    pipeline never raises on a per-lead failure.
  * Latency and (when available) token usage are tracked for the run.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Callable, Optional

import httpx

from job_scout.agents.evidence import EvidenceAgent
from job_scout.agents.planner import build_search_plan
from job_scout.agents.red_team import RedTeamAgent
from job_scout.agents.report import ReportAgent
from job_scout.agents.scoring import ScoringAgent
from job_scout.agents.search import SearchAgent
from job_scout.agents.validation import ValidationAgent
from job_scout.llm import LLM
from job_scout.observability import MeteredLLM, estimate_cost_usd, get_logger
from job_scout.schemas import (
    CandidateProfile,
    JobReport,
    RejectionLogEntry,
    RunMetrics,
    RunRecord,
    SearchCriteria,
)
from job_scout.search_providers.base import SearchProvider

_log = get_logger("orchestrator")


class Orchestrator:
    def __init__(
        self,
        *,
        search_provider: SearchProvider,
        http_client: httpx.Client,
        llm: Optional[LLM] = None,
        llm_model: Optional[str] = None,
        clock: Optional[Callable[[], date]] = None,
        max_results_per_query: int = 10,
        user_agent: str = "ExecutiveJobScout/0.1 (+local-run)",
        http_timeout_seconds: int = 15,
    ) -> None:
        self._search_provider = search_provider
        self._http_client = http_client
        self._llm = llm
        self._llm_model = llm_model
        self._clock = clock or (lambda: datetime.now(timezone.utc).date())
        self._max_results_per_query = max_results_per_query
        self._user_agent = user_agent
        self._http_timeout = http_timeout_seconds

    def run(
        self,
        profile: CandidateProfile,
        criteria: SearchCriteria,
    ) -> RunRecord:
        started = time.monotonic()

        # Wrap any provided LLM in a meter so we can sum tokens/cost.
        metered_llm = MeteredLLM(self._llm) if self._llm is not None else None

        _log.info(
            "run_start",
            seniority=profile.seniority_level.value,
            target_titles=criteria.target_titles or profile.target_archetypes,
            max_age_days=criteria.max_age_days,
            min_match_score=criteria.min_match_score,
            llm_provider=getattr(self._llm, "name", None),
            llm_model=self._llm_model,
            search_provider=self._search_provider.name.value,
        )

        plan = build_search_plan(profile, criteria)
        _log.info("plan_built", query_count=len(plan.queries))

        search_agent = SearchAgent(
            self._search_provider,
            max_results_per_query=self._max_results_per_query,
        )
        leads = search_agent.run(plan)
        _log.info(
            "search_complete",
            leads=len(leads),
            errors=len(search_agent.errors),
        )

        validation_agent = ValidationAgent(
            client=self._http_client,
            user_agent=self._user_agent,
            timeout_seconds=self._http_timeout,
            clock=self._clock,
        )
        scoring_agent = ScoringAgent(llm=metered_llm, clock=self._clock)
        evidence_agent = EvidenceAgent()
        report_agent = ReportAgent()
        red_team = RedTeamAgent()

        rejection_log: list[RejectionLogEntry] = [
            RejectionLogEntry(stage="search", url=None, reason=f"{q!r}: {err}")
            for q, err in search_agent.errors
        ]

        accepted: list[JobReport] = []
        validated_count = 0

        for lead in leads:
            v = validation_agent.validate(lead, criteria)
            if v.rejection is not None:
                _log.info(
                    "lead_rejected",
                    stage="validation",
                    url=str(lead.url),
                    reason=v.rejection.reason,
                )
                rejection_log.append(v.rejection)
                continue
            assert v.job is not None
            validated_count += 1

            job = v.job

            # Two-pass scoring.
            #
            # Pass 1: deterministic only. Fast, no LLM calls, no rate-limit
            # risk. We use this to gate LLM usage on jobs that will be
            # surfaced — most validated leads score below the threshold and
            # would burn LLM tokens on rationales nobody sees.
            score = scoring_agent.score(
                job, profile, criteria, skip_llm=True,
            )

            # Pre-filter on the deterministic match score. Jobs that
            # already fall below the threshold can't be saved by an LLM
            # rationale — the Red Team would reject them anyway. Skip the
            # LLM call and record a pre-filter rejection.
            if score.match_score < criteria.min_match_score:
                _log.info(
                    "lead_rejected",
                    stage="score_prefilter",
                    url=str(job.canonical_url),
                    match_score=score.match_score,
                    threshold=criteria.min_match_score,
                )
                rejection_log.append(
                    RejectionLogEntry(
                        stage="score_prefilter",
                        url=str(job.canonical_url),
                        reason=(
                            f"match score below threshold "
                            f"({score.match_score} < {criteria.min_match_score})"
                        ),
                    )
                )
                continue

            # Pass 2: this job is worth surfacing. Upgrade its rationale
            # with the LLM (if one is configured).
            if self._llm is not None:
                score = scoring_agent.score(job, profile, criteria, skip_llm=False)

            evidence_bundle = evidence_agent.extract(
                job=job, profile=profile, criteria=criteria, score=score
            )
            assembly = report_agent.assemble(
                job=job, score=score, evidence=evidence_bundle
            )

            decision = red_team.evaluate(assembly, criteria)
            if not decision.accepted:
                _log.info(
                    "lead_rejected",
                    stage="red_team",
                    url=str(job.canonical_url),
                    reasons=decision.reasons,
                    match_score=score.match_score,
                )
                rejection_log.append(
                    RejectionLogEntry(
                        stage="red_team",
                        url=str(job.canonical_url),
                        reason="; ".join(decision.reasons),
                    )
                )
                continue

            # Attach the decision to the surfaced report for the UI.
            accepted.append(
                assembly.report.model_copy(update={"red_team": decision})
            )

        latency_ms = int((time.monotonic() - started) * 1000)

        # Compute LLM cost from accumulated token counts.
        tokens_in = metered_llm.tokens_in if metered_llm else 0
        tokens_out = metered_llm.tokens_out if metered_llm else 0
        cost_usd = estimate_cost_usd(
            provider=getattr(self._llm, "name", None),
            model=self._llm_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        # Sort surfaced reports by a combined match+urgency priority
        # (criteria.prioritize_urgent uses urgency to break ties).
        accepted.sort(
            key=lambda r: (
                -r.score.match_score,
                -(r.score.urgency_score if criteria.prioritize_urgent else 0),
            )
        )

        record = RunRecord(
            criteria=criteria,
            plan=plan,
            llm_provider=getattr(self._llm, "name", None),
            llm_model=self._llm_model,
            search_provider=self._search_provider.name.value,
            metrics=RunMetrics(
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                discovered=len(leads),
                validated=validated_count,
                surfaced=len(accepted),
            ),
            rejection_log=rejection_log,
            reports=accepted,
        )

        _log.info(
            "run_complete",
            run_id=str(record.id),
            latency_ms=latency_ms,
            discovered=len(leads),
            validated=validated_count,
            surfaced=len(accepted),
            rejected=len(rejection_log),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )
        return record
