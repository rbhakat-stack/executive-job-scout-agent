"""Planner Agent.

Builds a `SearchPlan` from a `CandidateProfile` + `SearchCriteria` using
deterministic templates. No LLM involvement — by design, so the query set
is reproducible, debuggable, and not subject to model drift.

Strategies covered (see `docs/ARCHITECTURE.md`):
  1. EXACT_TITLE        - the user's target titles
  2. ADJACENT_TITLE     - alternative titles drawn from the profile
  3. INDUSTRY_AND_ROLE  - title x industry combinations
  4. COMPANY_AND_ROLE   - title x preferred-company combinations
  5. SKILL_AND_ROLE     - key skill x title combinations
  6. ATS_SCOPED         - site: filter for Greenhouse, Lever, Workday, etc.
  7. URGENT_HIRING      - 'urgently hiring' tag (only if criteria.prioritize_urgent)

A per-strategy cap keeps the total query count bounded on rich profiles.
Excluded companies are appended to non-ATS queries as `-"<company>"`.
"""
from __future__ import annotations

from typing import Optional

from src.schemas import (
    CandidateProfile,
    SearchCriteria,
    SearchPlan,
    SearchQuery,
    SearchStrategy,
)

# Stable site: hosts for ATS-scoped queries. The Validation Agent does the
# authoritative ATS detection from response URLs; this list is just what we
# tell the search engine to scope to.
ATS_SITE_FILTERS: tuple[str, ...] = (
    "boards.greenhouse.io",
    "jobs.lever.co",
    "myworkdayjobs.com",
    "jobs.ashbyhq.com",
    "jobs.smartrecruiters.com",
    "icims.com",
)

# Per-strategy fan-out caps. Without these the planner can produce 100+
# queries on a rich profile, which is wasteful and rate-limit prone.
DEFAULT_QUERY_CAPS: dict[SearchStrategy, int] = {
    SearchStrategy.EXACT_TITLE: 4,
    SearchStrategy.ADJACENT_TITLE: 5,
    SearchStrategy.INDUSTRY_AND_ROLE: 5,
    SearchStrategy.COMPANY_AND_ROLE: 6,
    SearchStrategy.SKILL_AND_ROLE: 4,
    SearchStrategy.ATS_SCOPED: 6,
    SearchStrategy.URGENT_HIRING: 2,
}


def build_search_plan(
    profile: CandidateProfile,
    criteria: SearchCriteria,
    *,
    query_caps: Optional[dict[SearchStrategy, int]] = None,
) -> SearchPlan:
    caps = {**DEFAULT_QUERY_CAPS, **(query_caps or {})}
    queries: list[SearchQuery] = []

    # Resolve inputs. Criteria takes precedence over profile defaults.
    primary_titles = _dedupe(criteria.target_titles or profile.target_archetypes)
    adjacent_titles = _dedupe(
        [t for t in profile.title_equivalents if t.lower() not in {p.lower() for p in primary_titles}]
    )
    industries = _dedupe(criteria.preferred_industries or profile.industries)
    companies = _dedupe(criteria.preferred_companies)
    skills = _dedupe(profile.search_keywords + profile.ai_data_cloud_experience)

    # If we don't even know what titles the user is targeting, return an
    # empty plan with a note — the Search Agent will surface this as 0 hits.
    if not primary_titles and not adjacent_titles:
        return SearchPlan(
            queries=[],
            planner_notes=(
                "No target titles available from criteria or profile. "
                "Cannot build a search plan."
            ),
        )

    # 1. EXACT_TITLE
    for title in primary_titles[: caps[SearchStrategy.EXACT_TITLE]]:
        queries.append(
            SearchQuery(
                strategy=SearchStrategy.EXACT_TITLE,
                text=_phrase(title),
                rationale=f"Exact target title: {title}",
                expected_recency_days=criteria.max_age_days,
            )
        )

    # 2. ADJACENT_TITLE
    for title in adjacent_titles[: caps[SearchStrategy.ADJACENT_TITLE]]:
        queries.append(
            SearchQuery(
                strategy=SearchStrategy.ADJACENT_TITLE,
                text=_phrase(title),
                rationale=f"Adjacent title from profile: {title}",
                expected_recency_days=criteria.max_age_days,
            )
        )

    # 3. INDUSTRY_AND_ROLE - cartesian, capped
    for title, industry in _pairs(primary_titles, industries)[
        : caps[SearchStrategy.INDUSTRY_AND_ROLE]
    ]:
        queries.append(
            SearchQuery(
                strategy=SearchStrategy.INDUSTRY_AND_ROLE,
                text=f"{_phrase(title)} {_phrase(industry)}",
                rationale=f"Title + industry: {title} / {industry}",
                expected_recency_days=criteria.max_age_days,
            )
        )

    # 4. COMPANY_AND_ROLE
    for company, title in _pairs(companies, primary_titles)[
        : caps[SearchStrategy.COMPANY_AND_ROLE]
    ]:
        queries.append(
            SearchQuery(
                strategy=SearchStrategy.COMPANY_AND_ROLE,
                text=f"{_phrase(company)} {_phrase(title)}",
                rationale=f"Company + title: {company} / {title}",
                expected_recency_days=criteria.max_age_days,
            )
        )

    # 5. SKILL_AND_ROLE
    for skill, title in _pairs(skills, primary_titles)[
        : caps[SearchStrategy.SKILL_AND_ROLE]
    ]:
        queries.append(
            SearchQuery(
                strategy=SearchStrategy.SKILL_AND_ROLE,
                text=f"{_phrase(skill)} {_phrase(title)}",
                rationale=f"Skill + title: {skill} / {title}",
                expected_recency_days=criteria.max_age_days,
            )
        )

    # 6. ATS_SCOPED
    primary_industry = industries[0] if industries else None
    for site, title in _pairs(list(ATS_SITE_FILTERS), primary_titles)[
        : caps[SearchStrategy.ATS_SCOPED]
    ]:
        terms = [_phrase(title)]
        if primary_industry:
            terms.append(_phrase(primary_industry))
        queries.append(
            SearchQuery(
                strategy=SearchStrategy.ATS_SCOPED,
                text=f"site:{site} " + " ".join(terms),
                site_filter=site,
                rationale=f"ATS-scoped: {site} for {title}",
                expected_recency_days=criteria.max_age_days,
            )
        )

    # 7. URGENT_HIRING (opt-in via criteria.prioritize_urgent)
    if criteria.prioritize_urgent:
        for title in primary_titles[: caps[SearchStrategy.URGENT_HIRING]]:
            queries.append(
                SearchQuery(
                    strategy=SearchStrategy.URGENT_HIRING,
                    text=f'"urgently hiring" {_phrase(title)}',
                    rationale=f"Urgency signal for {title}",
                    expected_recency_days=criteria.max_age_days,
                )
            )

    # Append excluded-company filters to non-ATS queries.
    excluded = [f'-"{c}"' for c in criteria.excluded_companies if c.strip()]
    if excluded:
        suffix = " " + " ".join(excluded)
        queries = [
            q
            if q.strategy is SearchStrategy.ATS_SCOPED
            else q.model_copy(update={"text": q.text + suffix})
            for q in queries
        ]

    return SearchPlan(
        queries=queries,
        planner_notes=(
            f"Built {len(queries)} queries across "
            f"{len({q.strategy for q in queries})} strategies."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phrase(s: str) -> str:
    """Wrap multi-word strings in quotes so search engines treat as exact phrase."""
    s = s.strip()
    return f'"{s}"' if " " in s else s


def _dedupe(items: list[str]) -> list[str]:
    """Preserve order, drop blanks + case-insensitive duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        s = (raw or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _pairs(a: list[str], b: list[str]) -> list[tuple[str, str]]:
    """Cartesian product preserving the natural read order. Empty if either side is."""
    if not a or not b:
        return []
    return [(x, y) for x in a for y in b]
