"""Orchestrator end-to-end tests with FakeSearchProvider + httpx.MockTransport."""
from __future__ import annotations

import json
from datetime import date

import httpx

from job_scout.orchestrator import Orchestrator
from job_scout.schemas import (
    CandidateProfile,
    SearchCriteria,
    SearchProviderName,
    SeniorityLevel,
)
from job_scout.search_providers import FakeSearchProvider, SearchProviderResult

TODAY = date(2026, 5, 27)


def _profile() -> CandidateProfile:
    return CandidateProfile(
        resume_text_sha256="a" * 64,
        summary="Senior life-sciences tech leader.",
        seniority_level=SeniorityLevel.SVP,
        industries=["life sciences", "pharma"],
        functional_expertise=["AI strategy", "transformation"],
        technical_expertise=["AI", "data"],
        ai_data_cloud_experience=["GenAI", "data platforms"],
        transformation_themes=["AI transformation"],
        target_archetypes=["VP AI Transformation"],
        title_equivalents=["SVP Technology"],
    )


def _jsonld(title="VP AI Transformation", date_str="2026-05-22"):
    block = {
        "@context": "https://schema.org",
        "@type": "JobPosting",
        "title": title,
        "hiringOrganization": {"name": "Acme Bio"},
        "description": (
            "<p>Lead AI transformation across pharma. We are urgently hiring "
            "for this role to drive impact across data platforms in life "
            "sciences. We have multiple openings on our senior leadership "
            "team. Contact jane@acme.com to learn more.</p>"
        ),
    }
    if date_str:
        block["datePosted"] = date_str
    return (
        f'<html><head><script type="application/ld+json">{json.dumps(block)}'
        f'</script></head><body><h1>{title}</h1></body></html>'
    )


class TestOrchestrator:
    def test_full_pipeline_produces_surfaced_report(self):
        url = "https://boards.greenhouse.io/acme/jobs/1"
        search_provider = FakeSearchProvider(
            responder=lambda q: [
                SearchProviderResult(title="VP AI Transformation", url=url)
            ]
        )

        def handler(req):
            return httpx.Response(200, text=_jsonld())

        client = httpx.Client(transport=httpx.MockTransport(handler))

        orch = Orchestrator(
            search_provider=search_provider,
            http_client=client,
            clock=lambda: TODAY,
        )
        run = orch.run(_profile(), SearchCriteria(target_titles=["VP AI"]))

        assert run.metrics.surfaced == 1
        assert run.metrics.discovered >= 1
        assert run.metrics.validated >= 1
        assert run.reports[0].red_team is not None
        assert run.reports[0].red_team.accepted

    def test_rejected_leads_recorded_in_rejection_log(self):
        # Provider returns a 404 page; validation rejects.
        url = "https://example.com/job/dead"
        search_provider = FakeSearchProvider(
            responder=lambda q: [SearchProviderResult(title="x", url=url)]
        )

        def handler(req):
            return httpx.Response(404, text="not found")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        orch = Orchestrator(
            search_provider=search_provider,
            http_client=client,
            clock=lambda: TODAY,
        )
        run = orch.run(_profile(), SearchCriteria(target_titles=["VP AI"]))
        assert run.metrics.surfaced == 0
        assert any("404" in r.reason for r in run.rejection_log)

    def test_sorts_surfaced_reports_by_match_then_urgency(self):
        # Two leads — one strong, one weak. Strong should come first.
        url1 = "https://boards.greenhouse.io/acme/jobs/strong"
        url2 = "https://boards.greenhouse.io/acme/jobs/middling"

        def search_responder(q):
            return [
                SearchProviderResult(title="strong", url=url1),
                SearchProviderResult(title="middling", url=url2),
            ]

        def handler(req):
            if "strong" in str(req.url):
                return httpx.Response(200, text=_jsonld(title="VP AI Transformation"))
            else:
                # Middling: title gives some credit, body has fewer overlaps.
                return httpx.Response(
                    200,
                    text=_jsonld(
                        title="VP Technology",
                        date_str=None,
                    ),
                )

        search_provider = FakeSearchProvider(responder=search_responder)
        client = httpx.Client(transport=httpx.MockTransport(handler))
        orch = Orchestrator(
            search_provider=search_provider,
            http_client=client,
            clock=lambda: TODAY,
        )
        run = orch.run(
            _profile(),
            SearchCriteria(target_titles=["VP AI"], min_match_score=0),
        )
        assert len(run.reports) >= 2
        # Match scores non-increasing.
        scores = [r.score.match_score for r in run.reports]
        assert scores == sorted(scores, reverse=True)

    def test_search_provider_error_recorded_not_raised(self):
        from job_scout.search_providers.base import SearchProviderError

        def responder(q):
            raise SearchProviderError("upstream down")

        search_provider = FakeSearchProvider(responder=responder)
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

        orch = Orchestrator(
            search_provider=search_provider,
            http_client=client,
            clock=lambda: TODAY,
        )
        run = orch.run(_profile(), SearchCriteria(target_titles=["VP AI"]))
        assert run.metrics.discovered == 0
        assert run.metrics.surfaced == 0
        assert any(r.stage == "search" for r in run.rejection_log)

    def test_metrics_capture_latency(self):
        url = "https://boards.greenhouse.io/acme/jobs/1"
        search_provider = FakeSearchProvider(
            responder=lambda q: [SearchProviderResult(title="x", url=url)]
        )

        def handler(req):
            return httpx.Response(200, text=_jsonld())

        client = httpx.Client(transport=httpx.MockTransport(handler))
        orch = Orchestrator(
            search_provider=search_provider,
            http_client=client,
            clock=lambda: TODAY,
        )
        run = orch.run(_profile(), SearchCriteria(target_titles=["VP AI"]))
        assert run.metrics.latency_ms >= 0

    def test_llm_not_called_for_below_threshold_jobs(self):
        """Two-pass scoring: jobs that fail the deterministic match
        threshold must NOT trigger an LLM call. Saves tokens + avoids
        rate limits on doomed leads."""
        from job_scout.llm import FakeLLM

        # Weak job: junior frontend role, no overlap with senior life-sci profile.
        weak_html = (
            '<html><head><script type="application/ld+json">'
            '{"@type":"JobPosting","title":"Junior Frontend Developer",'
            '"hiringOrganization":{"name":"FintechCo"},'
            '"description":"<p>We need a junior React dev to build dashboards. '
            'Bootcamp grads welcome. No leadership needed. Substantive body content.</p>",'
            '"datePosted":"2026-05-22"}'
            '</script></head><body><h1>Junior Frontend Developer</h1></body></html>'
        )

        url = "https://boards.greenhouse.io/fintechco/jobs/weak"
        search_provider = FakeSearchProvider(
            responder=lambda q: [SearchProviderResult(title="x", url=url)]
        )

        def handler(req):
            return httpx.Response(200, text=weak_html)

        client = httpx.Client(transport=httpx.MockTransport(handler))

        # Empty LLM queue: any LLM call would raise RuntimeError.
        # The two-pass flow must NOT call the LLM because the weak job
        # falls below the threshold after the deterministic pass.
        llm = FakeLLM(responses=[])

        orch = Orchestrator(
            search_provider=search_provider,
            http_client=client,
            llm=llm,
            clock=lambda: TODAY,
        )
        run = orch.run(
            _profile(),
            SearchCriteria(target_titles=["VP AI"], min_match_score=60),
        )

        # Job was rejected at the score_prefilter stage (not red_team).
        assert run.metrics.surfaced == 0
        assert llm.calls == [], (
            "LLM must not be called for jobs below the deterministic threshold"
        )
        assert any(
            r.stage == "score_prefilter" for r in run.rejection_log
        ), f"expected a score_prefilter rejection; got {[r.stage for r in run.rejection_log]}"
