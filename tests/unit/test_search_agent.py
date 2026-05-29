"""Search Agent + provider tests."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from job_scout.agents.planner import build_search_plan
from job_scout.agents.search import SearchAgent, detect_ats, is_likely_non_job
from job_scout.schemas import (
    ATS,
    CandidateProfile,
    SearchCriteria,
    SearchPlan,
    SearchProviderName,
    SearchQuery,
    SearchStrategy,
    SeniorityLevel,
)
from job_scout.search_providers import (
    FakeSearchProvider,
    SearchProviderError,
    SearchProviderResult,
    TavilySearchProvider,
)


# ---------------------------------------------------------------------------
# detect_ats
# ---------------------------------------------------------------------------

class TestDetectAts:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://boards.greenhouse.io/acme/jobs/123", ATS.GREENHOUSE),
            ("https://jobs.lever.co/acme/job/abc", ATS.LEVER),
            ("https://acme.wd5.myworkdayjobs.com/en-US/X/job/Y", ATS.WORKDAY),
            ("https://jobs.ashbyhq.com/acme/job-id", ATS.ASHBY),
            ("https://jobs.smartrecruiters.com/Acme/123", ATS.SMARTRECRUITERS),
            ("https://acme.icims.com/jobs/123/foo", ATS.ICIMS),
            ("https://www.example.com/careers/123", None),
            ("not a url", None),
            ("", None),
        ],
    )
    def test_detect(self, url: str, expected):
        assert detect_ats(url) == expected


class TestIsLikelyNonJob:
    """Pre-filter that drops obvious non-job URLs before validation."""

    @pytest.mark.parametrize(
        "url",
        [
            # Articles / blogs / insights
            "https://www.deloitte.com/us/en/insights/topics/digital-transformation",
            "https://www.ibm.com/think/topics/ai-transformation",
            "https://blog.example.com/blog/role-of-a-digital-leader",
            "https://example.com/articles/managing-director-vs-director",
            # Dictionaries / definitions
            "https://en.wikipedia.org/wiki/Partner_(business_rank)",
            "https://dictionary.cambridge.org/us/dictionary/english/senior-partner",
            "https://definitions.lsd.law/senior-partner",
            # Courses / programs / certifications
            "https://www.coursera.org/specializations/digital-transformation-leader",
            "https://execonline.hms.harvard.edu/senior-life-sciences-leaders-program",
            "https://www.usaii.org/artificial-intelligence-certifications/certified-ai-transformation-leader/",
            "https://www.cbtnuggets.com/it-training/microsoft-365/ab-731",
            # Articles in unusual paths
            "https://aws.amazon.com/what-is-data-strategy/",
            "https://example.com/how-to-write-a-managing-director-job-description",
            "https://example.com/guide-to-becoming-a-cdo",
            "https://crummer.rollins.edu/resources/executive-director-vs-managing-director/",
            "https://www.jdsupra.com/legalnews/senior-partners-are-the-golden-ticket-for-law-firms/",
            "https://business.linkedin.com/en-in/talent-solutions/resources/talent-acquisition/job-descriptions/managing-director-job-description-template-linkedin/",
            # PDFs and Office files
            "https://example.com/wp-content/uploads/2024/02/BIO-EA_Admin-as-posted-1.pdf",
            "https://example.com/jobs/some-job.pdf",
            # Social / videos
            "https://www.linkedin.com/in/janedoe",
            "https://www.linkedin.com/company/the-digital-transformation-leaders",
            "https://www.youtube.com/watch?v=abc",
            "https://medium.com/some-article",
            # Bot-walled job aggregators
            "https://www.indeed.com/q-Digital-Transformation-Leader-jobs.html",
            "https://www.ziprecruiter.com/Jobs/Digital-Transformation-Leader",
            # Topical landing pages
            "https://example.com/topics/ai",
            "https://example.com/category/leadership",
        ],
    )
    def test_obvious_non_jobs_are_filtered(self, url):
        assert is_likely_non_job(url) is True, f"should have been filtered: {url}"

    @pytest.mark.parametrize(
        "url",
        [
            # Real ATS job pages
            "https://boards.greenhouse.io/acme/jobs/123",
            "https://job-boards.greenhouse.io/10xgenomics/jobs/7537471",
            "https://jobs.lever.co/patsnap/abc-123",
            "https://jobs.ashbyhq.com/openai/job-id",
            "https://acme.wd5.myworkdayjobs.com/en-US/External/job/foo",
            "https://acme.icims.com/jobs/123/some-role",
            # Custom company career pages
            "https://careers.pfizer.com/job/12345",
            "https://www.databricks.com/company/careers/some-role",
            "https://www.example.com/careers/director-of-ai",
            # LinkedIn JOBS (different from /company/ or /in/)
            "https://www.linkedin.com/jobs/view/4123456789",
        ],
    )
    def test_legit_job_urls_pass(self, url):
        assert is_likely_non_job(url) is False, f"should have passed: {url}"

    def test_positive_pattern_overrides_negative(self):
        # An /insights/ path that also contains /jobs/ survives.
        assert is_likely_non_job(
            "https://example.com/insights/careers/jobs/director"
        ) is False

    def test_empty_url(self):
        assert is_likely_non_job("") is True


# ---------------------------------------------------------------------------
# FakeSearchProvider
# ---------------------------------------------------------------------------

def _result(
    url: str = "https://boards.greenhouse.io/acme/jobs/1",
    title: str = "VP AI",
    snippet: str = "Lead AI transformation",
    published_date: datetime | None = None,
) -> SearchProviderResult:
    return SearchProviderResult(
        title=title, url=url, snippet=snippet, published_date=published_date
    )


class TestFakeSearchProvider:
    def test_returns_mapped_results(self):
        p = FakeSearchProvider(results_by_query={"VP AI": [_result()]})
        out = p.search("VP AI")
        assert len(out) == 1
        assert out[0].title == "VP AI"

    def test_unknown_query_returns_empty(self):
        p = FakeSearchProvider(results_by_query={"x": [_result()]})
        assert p.search("nothing matches") == []

    def test_responder_mode(self):
        p = FakeSearchProvider(
            responder=lambda q: [_result(title=f"hit for {q}")]
        )
        out = p.search("VP AI")
        assert out[0].title == "hit for VP AI"

    def test_max_results_caps_output(self):
        p = FakeSearchProvider(
            results_by_query={
                "q": [_result(url=f"https://example.com/job/{i}") for i in range(5)]
            }
        )
        assert len(p.search("q", max_results=3)) == 3

    def test_must_provide_exactly_one_mode(self):
        with pytest.raises(ValueError):
            FakeSearchProvider()  # type: ignore[call-arg]
        with pytest.raises(ValueError):
            FakeSearchProvider(
                results_by_query={"x": []},
                responder=lambda q: [],
            )


# ---------------------------------------------------------------------------
# SearchAgent
# ---------------------------------------------------------------------------

def _profile() -> CandidateProfile:
    return CandidateProfile(
        resume_text_sha256="a" * 64,
        summary="Senior leader.",
        seniority_level=SeniorityLevel.SVP,
        target_archetypes=["VP AI"],
        industries=["life sciences"],
    )


def _plan(text: str = '"VP AI"') -> SearchPlan:
    return SearchPlan(
        queries=[
            SearchQuery(
                strategy=SearchStrategy.EXACT_TITLE,
                text=text,
                expected_recency_days=14,
            )
        ]
    )


class TestSearchAgent:
    def test_emits_raw_leads_for_results(self):
        provider = FakeSearchProvider(
            results_by_query={
                '"VP AI"': [
                    _result(url="https://boards.greenhouse.io/acme/jobs/1"),
                    _result(
                        url="https://jobs.lever.co/acme/job/2",
                        title="VP AI Strategy",
                    ),
                ]
            }
        )
        agent = SearchAgent(provider)
        leads = agent.run(_plan())
        assert len(leads) == 2
        assert leads[0].ats_guess is ATS.GREENHOUSE
        assert leads[1].ats_guess is ATS.LEVER
        assert all(l.source_provider is SearchProviderName.FAKE for l in leads)
        assert all(l.source_query.text == '"VP AI"' for l in leads)

    def test_dedups_across_queries_by_url(self):
        url = "https://boards.greenhouse.io/acme/jobs/1"
        provider = FakeSearchProvider(
            results_by_query={
                '"VP AI"': [_result(url=url)],
                '"Chief Digital Officer"': [_result(url=url, title="CDO")],
            }
        )
        plan = SearchPlan(
            queries=[
                SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text='"VP AI"'),
                SearchQuery(
                    strategy=SearchStrategy.EXACT_TITLE,
                    text='"Chief Digital Officer"',
                ),
            ]
        )
        agent = SearchAgent(provider)
        leads = agent.run(plan)
        assert len(leads) == 1  # second hit was a dup

    def test_provider_error_on_one_query_does_not_abort_run(self):
        def responder(q: str) -> list[SearchProviderResult]:
            if q == "BOOM":
                raise SearchProviderError("rate limited")
            return [_result()]

        provider = FakeSearchProvider(responder=responder)
        plan = SearchPlan(
            queries=[
                SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text="BOOM"),
                SearchQuery(strategy=SearchStrategy.EXACT_TITLE, text="OK"),
            ]
        )
        agent = SearchAgent(provider)
        leads = agent.run(plan)
        assert len(leads) == 1
        assert agent.errors == [("BOOM", "rate limited")]

    def test_propagates_search_engine_date(self):
        d = datetime(2026, 5, 20, tzinfo=timezone.utc)
        provider = FakeSearchProvider(
            results_by_query={'"VP AI"': [_result(published_date=d)]}
        )
        leads = SearchAgent(provider).run(_plan())
        assert leads[0].search_engine_date == d

    def test_pre_filter_drops_non_job_urls_and_counts_them(self):
        provider = FakeSearchProvider(
            results_by_query={
                '"VP AI"': [
                    _result(url="https://en.wikipedia.org/wiki/VP_AI"),
                    _result(url="https://example.com/wp-content/uploads/job.pdf"),
                    _result(url="https://boards.greenhouse.io/acme/jobs/123"),
                ]
            }
        )
        agent = SearchAgent(provider)
        leads = agent.run(_plan())
        assert len(leads) == 1
        assert "greenhouse.io" in str(leads[0].url)
        assert agent.pre_filtered == 2

    def test_end_to_end_with_planner(self):
        # Planner builds queries; provider returns canned results keyed by
        # the planner's quoted-phrase output. Confirms wire compatibility.
        profile = _profile()
        criteria = SearchCriteria(target_titles=["VP AI"])
        plan = build_search_plan(profile, criteria)
        assert plan.queries  # sanity

        provider = FakeSearchProvider(
            responder=lambda q: [_result(url=f"https://example.com/q/{hash(q) % 1000}")]
        )
        leads = SearchAgent(provider, max_results_per_query=1).run(plan)
        assert len(leads) > 0


# ---------------------------------------------------------------------------
# Tavily adapter (with httpx MockTransport)
# ---------------------------------------------------------------------------

class TestTavilyProvider:
    def test_parses_results(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == httpx.URL("https://api.tavily.com/search")
            body = request.read()
            import json
            payload = json.loads(body)
            assert payload["api_key"] == "fake-key"
            assert payload["query"] == "VP AI"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "VP AI Transformation",
                            "url": "https://boards.greenhouse.io/acme/jobs/1",
                            "content": "Lead AI.",
                            "published_date": "2026-05-20T00:00:00Z",
                        }
                    ]
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = TavilySearchProvider(api_key="fake-key", client=client)
        out = provider.search("VP AI")
        assert len(out) == 1
        assert out[0].title == "VP AI Transformation"
        assert out[0].published_date is not None
        assert out[0].published_date.year == 2026

    def test_non_200_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="rate limited")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = TavilySearchProvider(api_key="k", client=client)
        with pytest.raises(SearchProviderError, match="429"):
            provider.search("anything")

    def test_transport_error_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns failed")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = TavilySearchProvider(api_key="k", client=client)
        with pytest.raises(SearchProviderError, match="transport error"):
            provider.search("anything")

    def test_malformed_entries_are_skipped(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "ok", "url": "https://example.com/ok"},
                        {"title": "no url"},               # missing url - skipped
                        {"url": "not-a-valid-url"},        # invalid url - skipped
                        {"title": "ok2", "url": "https://example.com/ok2"},
                    ]
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = TavilySearchProvider(api_key="k", client=client)
        out = provider.search("q")
        assert {str(r.url) for r in out} == {
            "https://example.com/ok",
            "https://example.com/ok2",
        }

    def test_missing_api_key_raises_on_construction(self):
        with pytest.raises(SearchProviderError, match="required"):
            TavilySearchProvider(api_key="")
