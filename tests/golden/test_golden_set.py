"""Golden-set evaluation cases (Red Team rules).

Eight scenarios per the spec, each exercising the full validation → scoring
→ evidence → red team pipeline. Pass/fail here is the gate for "the system
is producing only real, recent, validated, citation-backed results."

Marked `golden` so they can be run alone via `pytest -m golden`.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import httpx
import pytest

from src.agents.evidence import EvidenceAgent
from src.agents.red_team import Reasons, RedTeamAgent
from src.agents.report import ReportAgent
from src.agents.scoring import ScoringAgent
from src.agents.validation import ValidationAgent
from src.schemas import (
    ATS,
    CandidateProfile,
    FreshnessLabel,
    RawJobLead,
    SearchCriteria,
    SearchProviderName,
    SearchQuery,
    SearchStrategy,
    SeniorityLevel,
)

pytestmark = pytest.mark.golden

TODAY = date(2026, 5, 27)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _profile() -> CandidateProfile:
    return CandidateProfile(
        resume_text_sha256="a" * 64,
        summary="Senior life-sciences tech leader with deep AI transformation.",
        seniority_level=SeniorityLevel.SVP,
        industries=["life sciences", "pharma"],
        functional_expertise=["AI strategy", "transformation"],
        technical_expertise=["AI", "data", "cloud"],
        ai_data_cloud_experience=["GenAI", "data platforms"],
        transformation_themes=["AI transformation"],
        target_archetypes=["VP AI Transformation", "Chief Digital Officer"],
        title_equivalents=["SVP Technology", "Head of AI"],
    )


def _lead(url: str) -> RawJobLead:
    return RawJobLead(
        title_guess="role",
        url=url,
        snippet="snippet",
        source_provider=SearchProviderName.TAVILY,
        source_query=SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text='"VP AI"'),
        ats_guess=ATS.GREENHOUSE,
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _validate(handler, lead: RawJobLead, criteria: SearchCriteria):
    return ValidationAgent(
        client=_client(handler), clock=lambda: TODAY
    ).validate(lead, criteria)


def _run_full_pipeline(handler, lead, criteria, red_team: RedTeamAgent | None = None):
    """Validation -> Scoring -> Evidence -> Report -> RedTeam, with mocked HTTP."""
    out = _validate(handler, lead, criteria)
    if out.rejection:
        return ("validation_rejected", out.rejection)

    job = out.job
    profile = _profile()
    score = ScoringAgent(clock=lambda: TODAY).score(job, profile, criteria)
    evidence = EvidenceAgent().extract(
        job=job, profile=profile, criteria=criteria, score=score
    )
    assembly = ReportAgent().assemble(job=job, score=score, evidence=evidence)
    rt = red_team or RedTeamAgent()
    decision = rt.evaluate(assembly, criteria)
    return ("decided", decision, assembly)


def _jsonld_page(
    *,
    title: str,
    company: str | None = "Acme Bio",
    date_posted: str | None = "2026-05-22",
    body: str = (
        "Lead AI transformation across pharma. We are urgently hiring "
        "for this role to drive impact across data platforms in life "
        "sciences. We have multiple openings on our senior leadership "
        "team. Contact jane@acme.com to learn more."
    ),
    extra_body: str = "",
) -> str:
    block: dict = {
        "@context": "https://schema.org",
        "@type": "JobPosting",
        "title": title,
        "description": f"<p>{body}{extra_body}</p>",
    }
    if company is not None:
        block["hiringOrganization"] = {"@type": "Organization", "name": company}
    if date_posted is not None:
        block["datePosted"] = date_posted
    return (
        f'<html><head><script type="application/ld+json">{json.dumps(block)}'
        f'</script></head><body><h1>{title}</h1></body></html>'
    )


# ---------------------------------------------------------------------------
# 1. Known active job -> accepted
# ---------------------------------------------------------------------------

class TestKnownActiveJob:
    def test_accepted_with_recent_freshness(self):
        def handler(req):
            return httpx.Response(200, text=_jsonld_page(title="VP AI Transformation"))

        outcome = _run_full_pipeline(
            handler,
            _lead("https://boards.greenhouse.io/acme/jobs/1"),
            SearchCriteria(),
        )
        assert outcome[0] == "decided"
        _, decision, assembly = outcome
        assert decision.accepted, decision.reasons
        assert assembly.report.job.freshness.label is FreshnessLabel.RECENT


# ---------------------------------------------------------------------------
# 2. Known expired job -> rejected at validation
# ---------------------------------------------------------------------------

class TestKnownExpiredJob:
    def test_rejected_at_validation_layer(self):
        # JSON-LD intact, but body says it's closed.
        def handler(req):
            return httpx.Response(
                200,
                text=_jsonld_page(
                    title="VP AI Transformation",
                    extra_body=" This position is no longer available.",
                ),
            )

        outcome = _run_full_pipeline(
            handler,
            _lead("https://boards.greenhouse.io/acme/jobs/expired"),
            SearchCriteria(),
        )
        assert outcome[0] == "validation_rejected"
        assert "expired" in outcome[1].reason


# ---------------------------------------------------------------------------
# 3. Duplicate job across multiple sources -> 2nd rejected at red team
# ---------------------------------------------------------------------------

class TestDuplicateAcrossSources:
    def test_second_lead_with_same_canonical_url_rejected(self):
        # Two different source URLs (different utm params) that resolve to
        # the same canonical URL via redirect/normalization.
        canonical_html = _jsonld_page(title="VP AI Transformation")

        def handler(req):
            # All requests return the same posting body.
            return httpx.Response(200, text=canonical_html)

        rt = RedTeamAgent()

        lead1 = _lead("https://boards.greenhouse.io/acme/jobs/1?utm_source=tw")
        lead2 = _lead("https://boards.greenhouse.io/acme/jobs/1?utm_source=li")

        o1 = _run_full_pipeline(handler, lead1, SearchCriteria(), red_team=rt)
        o2 = _run_full_pipeline(handler, lead2, SearchCriteria(), red_team=rt)

        assert o1[0] == "decided" and o1[1].accepted
        assert o2[0] == "decided" and not o2[1].accepted
        assert Reasons.DUPLICATE_IN_RUN in o2[1].reasons


# ---------------------------------------------------------------------------
# 4. Job with missing posting date -> UNKNOWN freshness, accepted if match high
# ---------------------------------------------------------------------------

class TestMissingPostingDate:
    def test_unknown_freshness_does_not_block(self):
        def handler(req):
            return httpx.Response(
                200, text=_jsonld_page(title="VP AI Transformation", date_posted=None)
            )

        outcome = _run_full_pipeline(
            handler,
            _lead("https://boards.greenhouse.io/acme/jobs/2"),
            SearchCriteria(),
        )
        assert outcome[0] == "decided"
        _, decision, assembly = outcome
        # Match is strong, so even without a date the job is accepted.
        assert decision.accepted, decision.reasons
        assert assembly.report.job.freshness.label is FreshnessLabel.UNKNOWN
        # Critically: the schema invariant means there's NO evidence object
        # for UNKNOWN, so it's structurally impossible to surface this as RECENT.
        assert assembly.report.job.freshness.evidence is None


# ---------------------------------------------------------------------------
# 5. Job with weak match -> rejected by red team threshold
# ---------------------------------------------------------------------------

class TestWeakMatch:
    def test_low_match_rejected(self):
        def handler(req):
            return httpx.Response(
                200,
                text=_jsonld_page(
                    title="Junior Frontend Developer",
                    company="FintechCo",
                    body=(
                        "We need a junior React developer to build dashboards. "
                        "Bootcamp grads welcome. No leadership experience needed. "
                        "Substantive body content to pass length check."
                    ),
                ),
            )

        outcome = _run_full_pipeline(
            handler,
            _lead("https://boards.greenhouse.io/fintechco/jobs/weak"),
            SearchCriteria(min_match_score=60),
        )
        assert outcome[0] == "decided"
        _, decision, _ = outcome
        assert not decision.accepted
        assert any(Reasons.MATCH_BELOW_THRESHOLD in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# 6. Job with strong match -> accepted (sanity reverse of #5)
# ---------------------------------------------------------------------------

class TestStrongMatch:
    def test_high_match_accepted(self):
        def handler(req):
            return httpx.Response(200, text=_jsonld_page(title="VP AI Transformation"))

        outcome = _run_full_pipeline(
            handler,
            _lead("https://boards.greenhouse.io/acme/jobs/strong"),
            SearchCriteria(min_match_score=60),
        )
        assert outcome[0] == "decided"
        _, decision, assembly = outcome
        assert decision.accepted, decision.reasons
        assert assembly.report.score.match_score >= 60


# ---------------------------------------------------------------------------
# 7. Job with misleading title -> rejected by score threshold (body wins)
# ---------------------------------------------------------------------------

class TestMisleadingTitle:
    def test_vp_ai_marketing_coordinator_rejected(self):
        # Title contains 'VP AI' but body is a junior marketing role.
        def handler(req):
            return httpx.Response(
                200,
                text=_jsonld_page(
                    title="VP AI Marketing Coordinator",
                    company="GenericCo",
                    body=(
                        "Manage marketing emails and run AI-themed campaigns. "
                        "Entry-level position, no leadership responsibility. "
                        "Substantive body content to pass length check."
                    ),
                ),
            )

        outcome = _run_full_pipeline(
            handler,
            _lead("https://boards.greenhouse.io/genericco/jobs/misleading"),
            SearchCriteria(min_match_score=60),
        )
        assert outcome[0] == "decided"
        _, decision, assembly = outcome
        # Title alone can earn ~15 points; without industry/functional/tech
        # overlap from the body, score stays below 60.
        assert not decision.accepted
        assert assembly.report.score.match_score < 60
        assert any(Reasons.MATCH_BELOW_THRESHOLD in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# 8. Job behind ATS redirect (302 -> live posting page)
# ---------------------------------------------------------------------------

class TestAtsRedirect:
    def test_redirect_followed_and_canonical_url_used(self):
        # First request returns a redirect; second returns the JSON-LD page.
        # httpx.MockTransport handles redirects automatically when the
        # response carries a Location header and a 3xx status.
        original_url = "https://acme.com/careers/123"
        canonical_url = "https://boards.greenhouse.io/acme/jobs/redirect-target"

        def handler(req):
            if str(req.url) == original_url:
                return httpx.Response(
                    302,
                    headers={"location": canonical_url},
                )
            return httpx.Response(
                200,
                text=_jsonld_page(title="VP AI Transformation"),
            )

        outcome = _run_full_pipeline(
            handler,
            _lead(original_url),
            SearchCriteria(),
        )
        assert outcome[0] == "decided"
        _, decision, assembly = outcome
        assert decision.accepted, decision.reasons
        # The validated job's canonical URL is the post-redirect target.
        assert str(assembly.report.job.canonical_url) == canonical_url
        assert assembly.report.job.validation.redirected is True
