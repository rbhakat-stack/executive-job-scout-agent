"""Search Agent.

Executes a `SearchPlan` against a `SearchProvider` and returns a flat list
of `RawJobLead`. Performs URL-level de-duplication, ATS detection, and a
URL pre-filter that drops obvious non-job pages (articles, dictionaries,
landing pages, PDFs, social media) before they reach Validation.

Provider failures on a single query are recorded and skipped — they do not
abort the run, because partial results beat no results. Final de-duplication
(by content hash) is the Validation Agent's job.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from src.schemas import ATS, RawJobLead, SearchPlan
from src.search_providers.base import SearchProvider, SearchProviderError

# URL host substring -> ATS detection. Order matters: more specific hosts
# (boards.greenhouse.io) before more general ones (greenhouse.io).
_ATS_HOST_RULES: tuple[tuple[str, ATS], ...] = (
    ("boards.greenhouse.io", ATS.GREENHOUSE),
    ("greenhouse.io",        ATS.GREENHOUSE),
    ("jobs.lever.co",        ATS.LEVER),
    ("lever.co",             ATS.LEVER),
    ("myworkdayjobs.com",    ATS.WORKDAY),
    ("workday.com",          ATS.WORKDAY),
    ("jobs.ashbyhq.com",     ATS.ASHBY),
    ("ashbyhq.com",          ATS.ASHBY),
    ("jobs.smartrecruiters.com", ATS.SMARTRECRUITERS),
    ("smartrecruiters.com",  ATS.SMARTRECRUITERS),
    ("icims.com",            ATS.ICIMS),
)


def detect_ats(url: str) -> Optional[ATS]:
    """Return the detected ATS for a URL, or None if no rule matches."""
    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    for pattern, ats in _ATS_HOST_RULES:
        if pattern in host:
            return ats
    return None


# URL path fragments that strongly suggest the page is NOT a job posting.
# Driven by real Tavily-result analysis: definitions, blog articles, course
# pages, sponsored content, whitepapers, and the like.
_NON_JOB_PATH_PATTERNS: tuple[str, ...] = (
    "/blog/", "/blogs/",
    "/insights/", "/insight/",
    "/articles/", "/article/",
    "/wiki/",
    "/dictionary/", "/glossary/", "/definitions/",
    "/topics/", "/topic/",
    "/specializations/", "/specialization/",
    "/category/", "/categories/",
    "/programs/", "/program/",
    "/courses/", "/course/",
    "/certifications/", "/certification/",
    "/training/", "/it-training/", "/learning/",
    "/sponsored/",
    "/think/",                          # ibm.com/think
    "/uploads/", "/wp-content/uploads/",
    "/whitepaper", "/whitepapers/",
    "/case-studies/", "/case-study/",
    "/research/",
    "/resources/", "/resource/",        # generic "resources" sections (articles, templates)
    "/reports/", "/report/",
    "/podcast/", "/podcasts/",
    "/news/",
    "/legalnews/",                      # jdsupra-style legal articles
    "/press/", "/press-release/", "/press-releases/",
    "/about/", "/about-us/",
    "/team/", "/our-team/", "/people/",
    "/perspectives/",
    "/publications/", "/publication/",
    "/library/",
    "/community/",                      # community.sap.com, etc.
    "/template/", "/templates/",        # job-description templates
    "/sample/", "/samples/",
    "/example/", "/examples/",
    "/what-is-", "/what-are-",          # "/what-is-data-strategy/" style
    "/how-to-", "/how-do-", "/howto-",
    "/guide-to-", "/guides-to-",
    "-vs-",                             # comparison articles ("director-vs-vp"), substring not anchored
)

# Domains that don't host job postings — articles, definitions, courses,
# social, video, opinion. Matched as substring of the host.
_NON_JOB_DOMAINS: tuple[str, ...] = (
    "wikipedia.org",
    "coursera.org", "edx.org", "udemy.com", "udacity.com",
    "cambridge.org", "merriam-webster.com", "dictionary.com", "investopedia.com",
    "hbr.org", "harvard.edu/programs",
    "youtube.com", "vimeo.com",
    "instagram.com", "facebook.com", "twitter.com", "x.com",
    "reddit.com", "medium.com", "substack.com",
    "linkedin.com/company",            # company pages (NOT linkedin.com/jobs/)
    "linkedin.com/in/",                # personal profiles
    "linkedin.com/posts/",             # posts/articles
    "linkedin.com/pulse/",             # LinkedIn articles
    "lin.linkedin.com",                # Indian LinkedIn jobs - usually login-walled
    "definitions.lsd.law",
    "lsd.law",                          # parent domain
    "legal-resources.uslegalforms.com",
    "lawcrossing.com",
    "jdsupra.com",                      # legal news / articles
    "epscientific.com",                 # epmscientific.com bot-walls aggressively
    "execonline.",                      # Harvard / MIT executive-education portals
    "executive.stanford.edu",
    "exec.wharton.upenn.edu",
    "cbtnuggets.com",                   # IT certification training
    "usaii.org",                        # United States Artificial Intelligence Institute (certs, not jobs)
    "coursera.org",                     # already in above; keep for clarity
    "business.linkedin.com",            # LinkedIn talent-solutions content (not jobs)
    "linkedin.com/learning",            # LinkedIn Learning courses
    "talent.com/blog",                  # talent.com main domain has real jobs; blog doesn't
)

# Hosts known to bot-block aggressively (403/429 every time). Saves
# validation cycles + retry storms.
_BOT_BLOCKED_HOSTS: tuple[str, ...] = (
    "indeed.com",
    "ziprecruiter.com",
    "glassdoor.com",
    "sa.jooble.org",
)

# File extensions we can't parse (PDFs, Office docs).
_NON_HTML_EXTENSIONS: tuple[str, ...] = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz",
)

# Positive: URL path containing any of these is *probably* a job posting.
# Used to override the non-job-pattern reject when both match.
_LIKELY_JOB_PATTERNS: tuple[str, ...] = (
    "/job/", "/jobs/",
    "/careers/", "/career/",
    "/vacancy/", "/vacancies/",
    "/position/", "/positions/",
    "/opening/", "/openings/",
    "/posting/", "/postings/",
    "/req/", "/requisition/",
    "/role/", "/roles/",
    "/opportunity/", "/opportunities/",
    "/apply/",
)


def is_likely_non_job(url: str) -> bool:
    """Heuristic pre-filter: True when the URL is almost certainly NOT a job.

    Applied in the Search Agent before HTTP fetch so we don't burn validation
    cycles on articles, dictionaries, PDFs, social media, and bot-walled
    hosts that real-world Tavily results constantly include.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return True
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if not host:
        return True

    # Non-HTML files we can't parse.
    if any(path.endswith(ext) for ext in _NON_HTML_EXTENSIONS):
        return True

    # Hard-rejected domains (articles, definitions, social, etc.).
    for dom in _NON_JOB_DOMAINS:
        if dom in host or dom in (host + path):
            return True

    # Hard-rejected hosts (bot walls).
    for h in _BOT_BLOCKED_HOSTS:
        if h in host:
            return True

    # Path heuristic: if it screams "article" AND nothing screams "job",
    # reject. The positive override means "/blog/jobs/" or "/insights/careers/"
    # (rare but possible) survives.
    has_non_job = any(p in path for p in _NON_JOB_PATH_PATTERNS)
    has_job = any(p in path for p in _LIKELY_JOB_PATTERNS)
    if has_non_job and not has_job:
        return True

    return False


class SearchAgentError(Exception):
    pass


class SearchAgent:
    """Executes a search plan against a provider and emits RawJobLeads."""

    def __init__(
        self,
        provider: SearchProvider,
        *,
        max_results_per_query: int = 10,
    ) -> None:
        self._provider = provider
        self._max_per_q = max_results_per_query
        # Recorded errors: list of (query_text, error_message). Surfaced
        # through the orchestrator's rejection log.
        self.errors: list[tuple[str, str]] = []

    def run(self, plan: SearchPlan) -> list[RawJobLead]:
        seen_urls: set[str] = set()
        leads: list[RawJobLead] = []
        self.pre_filtered = 0  # count of URLs dropped by is_likely_non_job

        for query in plan.queries:
            try:
                results = self._provider.search(
                    query.text, max_results=self._max_per_q
                )
            except SearchProviderError as e:
                self.errors.append((query.text, str(e)))
                continue

            for r in results:
                url_s = str(r.url)
                if url_s in seen_urls:
                    continue
                seen_urls.add(url_s)

                # Pre-filter: drop obvious non-jobs BEFORE validation gets
                # them. Saves validation cycles + keeps the rejection log
                # focused on real failures.
                if is_likely_non_job(url_s):
                    self.pre_filtered += 1
                    continue

                leads.append(
                    RawJobLead(
                        title_guess=(r.title or "(unknown)").strip()[:200],
                        url=r.url,
                        snippet=r.snippet,
                        source_provider=self._provider.name,
                        source_query=query,
                        ats_guess=detect_ats(url_s),
                        search_engine_date=r.published_date,
                    )
                )

        return leads
