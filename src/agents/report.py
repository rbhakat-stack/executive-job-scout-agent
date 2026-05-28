"""Report Agent.

Assembles a `JobReport` from the validated job, the score, and the
evidence bundle. Performs only structural assembly — the substantive
quality gate (citations cover every non-zero feature, freshness honors
criteria, rationale is substantive, etc.) is the Red Team Agent's job
in M7.

The agent also reports `coverage_gaps` so callers can route gap-bearing
reports to the Red Team's rejection path before they reach the UI.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.evidence import coverage_gaps
from src.schemas import EvidenceBundle, JobReport, ScoreResult, ValidatedJob


@dataclass
class ReportAssembly:
    """Output of the Report Agent: the report + any structural diagnostics."""

    report: JobReport
    coverage_gaps: list[str]   # claim_keys referenced by features but missing from evidence


class ReportAgent:
    def assemble(
        self,
        *,
        job: ValidatedJob,
        score: ScoreResult,
        evidence: EvidenceBundle,
    ) -> ReportAssembly:
        report = JobReport(job=job, score=score, evidence=evidence)
        gaps = coverage_gaps(score, evidence)
        return ReportAssembly(report=report, coverage_gaps=gaps)
