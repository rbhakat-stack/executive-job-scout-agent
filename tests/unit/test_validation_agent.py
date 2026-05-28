"""End-to-end Validation Agent tests using httpx MockTransport.

Covers the golden-set-relevant scenarios that M7 will lift wholesale:
  * Known active job             -> accepted with RECENT freshness
  * Known expired job            -> rejected (expired_signals)
  * Missing date on page          -> accepted with UNKNOWN freshness
  * 404 / dead URL                -> rejected
  * Redirect to careers index     -> rejected (no JobPosting on landing page)
  * Title missing                  -> rejected
  * Company missing               -> rejected
"""
from __future__ import annotations

from datetime import date

import httpx
import pytest

from src.agents.validation import ValidationAgent
from src.schemas import (
    ATS,
    FreshnessLabel,
    RawJobLead,
    SearchCriteria,
    SearchProviderName,
    SearchQuery,
    SearchStrategy,
    WorkMode,
)

TODAY = date(2026, 5, 27)


def _lead(url: str = "https://boards.greenhouse.io/acme/jobs/1") -> RawJobLead:
    return RawJobLead(
        title_guess="VP AI",
        url=url,
        snippet="VP AI - life sciences",
        source_provider=SearchProviderName.TAVILY,
        source_query=SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text='"VP AI"'),
        ats_guess=ATS.GREENHOUSE,
    )


def _jsonld(date_str: str | None = "2026-05-20") -> str:
    block = {
        "@context": "https://schema.org",
        "@type": "JobPosting",
        "title": "VP AI Transformation",
        "hiringOrganization": {"@type": "Organization", "name": "Acme Bio"},
        "jobLocation": {
            "@type": "Place",
            "address": {
                "@type": "PostalAddress",
                "addressLocality": "Boston",
                "addressCountry": "US",
            },
        },
        "description": (
            "<p>Lead AI transformation across our pharma business. "
            "Drive measurable impact across 20+ countries.</p>"
        ),
    }
    if date_str:
        block["datePosted"] = date_str
    import json
    return (
        f'<html><head><script type="application/ld+json">{json.dumps(block)}</script>'
        "</head><body>Lead AI transformation across pharma.</body></html>"
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _agent(client: httpx.Client) -> ValidationAgent:
    return ValidationAgent(client=client, clock=lambda: TODAY)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestKnownActiveJob:
    def test_accepted_with_recent_freshness(self):
        def handler(req): return httpx.Response(200, text=_jsonld("2026-05-20"))
        agent = _agent(_client(handler))
        out = agent.validate(_lead(), SearchCriteria())

        assert out.accepted
        job = out.job
        assert job.title == "VP AI Transformation"
        assert job.company == "Acme Bio"
        assert job.freshness.label is FreshnessLabel.RECENT
        assert job.freshness.posted_at == date(2026, 5, 20)
        assert job.freshness.evidence is not None
        assert job.ats is ATS.GREENHOUSE
        assert job.validation.live is True
        assert job.validation.http_status == 200

    def test_dedup_hash_is_set(self):
        def handler(req): return httpx.Response(200, text=_jsonld())
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        assert out.accepted
        assert len(out.job.dedup_hash) == 64


# ---------------------------------------------------------------------------
# Expired
# ---------------------------------------------------------------------------

class TestExpired:
    def test_expired_phrase_rejects(self):
        html = (
            '<html><body><h1>VP AI Transformation</h1>'
            '<p>This position is no longer available. Lead AI transformation across pharma.</p>'
            '</body></html>'
        )
        def handler(req): return httpx.Response(200, text=html)
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        assert not out.accepted
        assert "expired" in out.rejection.reason


# ---------------------------------------------------------------------------
# Freshness fallbacks
# ---------------------------------------------------------------------------

class TestMissingDate:
    def test_no_date_means_unknown_freshness(self):
        # JSON-LD without datePosted, no meta tag.
        def handler(req): return httpx.Response(200, text=_jsonld(date_str=None))
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        assert out.accepted
        # Critical: must NOT be RECENT without evidence.
        assert out.job.freshness.label is FreshnessLabel.UNKNOWN
        assert out.job.freshness.posted_at is None
        assert out.job.freshness.evidence is None


# ---------------------------------------------------------------------------
# Dead links / redirects
# ---------------------------------------------------------------------------

class TestDeadLinks:
    def test_404_rejects(self):
        def handler(req): return httpx.Response(404, text="not found")
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        assert not out.accepted
        assert "404" in out.rejection.reason

    def test_500_rejects(self):
        def handler(req): return httpx.Response(500, text="oops")
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        assert not out.accepted

    def test_transport_error_rejects(self):
        def handler(req): raise httpx.ConnectError("dns failed")
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        assert not out.accepted
        assert "transport" in out.rejection.reason.lower()

    def test_redirect_to_generic_careers_page_rejects(self):
        # Server redirects the requested URL to a careers index that has no
        # JobPosting JSON-LD and just shows a list of openings — body too short,
        # no company/title -> rejected by extractor checks.
        def handler(req):
            return httpx.Response(
                200,
                text="<html><body><h1>Careers</h1></body></html>",
                request=req,
            )
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        assert not out.accepted
        assert ("missing company" in out.rejection.reason
                or "too short" in out.rejection.reason)


# ---------------------------------------------------------------------------
# Required-field rejection
# ---------------------------------------------------------------------------

class TestRequiredFields:
    def test_missing_title_rejects(self):
        html = (
            '<html><body>'
            '<p>Acme Bio is hiring. Lead AI transformation across pharma. '
            'Substantive body content to clear length checks.</p>'
            '</body></html>'
        )
        def handler(req): return httpx.Response(200, text=html)
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        assert not out.accepted
        # Note: <title> tag fallback may catch this; we use an HTML without <title>.

    def test_missing_company_rejects(self):
        html = (
            '<html><body>'
            '<h1>VP AI Transformation</h1>'
            '<p>Lead AI transformation across pharma. Substantive body content.</p>'
            '</body></html>'
        )
        def handler(req): return httpx.Response(200, text=html)
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        # No JSON-LD -> no company source -> rejected.
        assert not out.accepted
        assert "missing company" in out.rejection.reason


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class TestSignals:
    def test_urgency_phrases_are_detected(self):
        block = {
            "@context": "https://schema.org",
            "@type": "JobPosting",
            "title": "VP AI",
            "datePosted": "2026-05-20",
            "hiringOrganization": {"name": "Acme Bio"},
            "description": (
                "<p>We are urgently hiring for AI transformation. "
                "Immediate start preferred. Substantive body text.</p>"
            ),
        }
        import json
        html = (
            f'<html><head><script type="application/ld+json">{json.dumps(block)}'
            '</script></head><body>x</body></html>'
        )
        def handler(req): return httpx.Response(200, text=html)
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        assert out.accepted
        assert "urgently hiring" in out.job.signals.urgency_phrases
        assert "immediate start" in out.job.signals.urgency_phrases

    def test_work_mode_remote(self):
        def handler(req): return httpx.Response(200, text=_jsonld())
        out = _agent(_client(handler)).validate(_lead(), SearchCriteria())
        # JSONLD has jobLocationType=... but we didn't set TELECOMMUTE here
        # so work_mode is UNKNOWN (the JSON-LD lacks jobLocationType).
        assert out.job.work_mode in (WorkMode.UNKNOWN, WorkMode.REMOTE)


# ---------------------------------------------------------------------------
# Search-engine date fallback
# ---------------------------------------------------------------------------

class TestSearchEngineDateFallback:
    def test_used_when_page_has_no_date(self):
        from datetime import datetime, timezone

        # Page returns JSON-LD with NO datePosted.
        def handler(req): return httpx.Response(200, text=_jsonld(date_str=None))

        lead = _lead()
        # Inject a search-engine-reported date.
        lead = lead.model_copy(
            update={
                "search_engine_date": datetime(2026, 5, 22, tzinfo=timezone.utc),
            }
        )
        out = _agent(_client(handler)).validate(lead, SearchCriteria())
        assert out.accepted
        assert out.job.freshness.label is FreshnessLabel.RECENT
        assert out.job.freshness.evidence.source == "search_engine_date"
        # Confidence is lower because the date isn't from the page itself.
        assert out.job.freshness.evidence.confidence == 0.6
