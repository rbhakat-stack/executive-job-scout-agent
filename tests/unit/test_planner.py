"""Planner Agent tests.

These pin the deterministic query templates. The planner has zero LLM calls,
so the tests are exact-string assertions on generated queries.
"""
from __future__ import annotations

import pytest

from src.agents.planner import (
    ATS_SITE_FILTERS,
    DEFAULT_QUERY_CAPS,
    build_search_plan,
)
from src.schemas import (
    CandidateProfile,
    SearchCriteria,
    SearchStrategy,
    SeniorityLevel,
)


def _profile(**overrides) -> CandidateProfile:
    base = dict(
        resume_text_sha256="a" * 64,
        summary="Senior life-sciences tech leader.",
        seniority_level=SeniorityLevel.SVP,
        industries=["life sciences", "pharma"],
        target_archetypes=["VP AI Transformation", "Chief Digital Officer"],
        title_equivalents=["SVP Technology", "Head of AI"],
        search_keywords=["AI transformation"],
        ai_data_cloud_experience=["GenAI strategy", "data platforms"],
    )
    base.update(overrides)
    return CandidateProfile(**base)


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_no_titles_anywhere_returns_empty_plan(self):
        profile = _profile(target_archetypes=[], title_equivalents=[])
        plan = build_search_plan(profile, SearchCriteria())
        assert plan.queries == []
        assert "No target titles" in plan.planner_notes


class TestExactTitle:
    def test_uses_criteria_target_titles_when_provided(self):
        profile = _profile()
        criteria = SearchCriteria(target_titles=["Managing Director"])
        plan = build_search_plan(profile, criteria)

        exact = [q for q in plan.queries if q.strategy is SearchStrategy.EXACT_TITLE]
        assert len(exact) == 1
        assert exact[0].text == '"Managing Director"'

    def test_falls_back_to_profile_archetypes(self):
        profile = _profile()
        plan = build_search_plan(profile, SearchCriteria())

        exact = [q for q in plan.queries if q.strategy is SearchStrategy.EXACT_TITLE]
        texts = {q.text for q in exact}
        assert '"VP AI Transformation"' in texts
        assert '"Chief Digital Officer"' in texts

    def test_single_word_title_is_not_quoted(self):
        profile = _profile(target_archetypes=["Partner"])
        plan = build_search_plan(profile, SearchCriteria())
        exact = [q for q in plan.queries if q.strategy is SearchStrategy.EXACT_TITLE]
        assert exact[0].text == "Partner"


class TestAdjacentTitle:
    def test_pulls_from_title_equivalents(self):
        profile = _profile()
        plan = build_search_plan(profile, SearchCriteria())

        adj = [q for q in plan.queries if q.strategy is SearchStrategy.ADJACENT_TITLE]
        texts = {q.text for q in adj}
        assert '"SVP Technology"' in texts
        assert '"Head of AI"' in texts

    def test_drops_titles_already_in_primary(self):
        # Move SVP Technology into target_titles; planner should not also
        # emit it as an adjacent query.
        profile = _profile()
        criteria = SearchCriteria(target_titles=["SVP Technology"])
        plan = build_search_plan(profile, criteria)

        adj_texts = {
            q.text
            for q in plan.queries
            if q.strategy is SearchStrategy.ADJACENT_TITLE
        }
        assert '"SVP Technology"' not in adj_texts


class TestIndustryAndRole:
    def test_pairs_titles_with_industries(self):
        profile = _profile(industries=["life sciences"])
        criteria = SearchCriteria(target_titles=["VP AI"])
        plan = build_search_plan(profile, criteria)

        pairs = [q for q in plan.queries if q.strategy is SearchStrategy.INDUSTRY_AND_ROLE]
        assert pairs[0].text == '"VP AI" "life sciences"'

    def test_no_industries_means_no_industry_queries(self):
        profile = _profile(industries=[])
        criteria = SearchCriteria(target_titles=["VP AI"], preferred_industries=[])
        plan = build_search_plan(profile, criteria)
        assert not [q for q in plan.queries if q.strategy is SearchStrategy.INDUSTRY_AND_ROLE]


class TestCompanyAndRole:
    def test_emits_only_when_companies_present(self):
        profile = _profile()
        plan = build_search_plan(profile, SearchCriteria())
        assert not [q for q in plan.queries if q.strategy is SearchStrategy.COMPANY_AND_ROLE]

    def test_quotes_company_and_title(self):
        plan = build_search_plan(
            _profile(),
            SearchCriteria(
                target_titles=["VP AI"],
                preferred_companies=["Pfizer"],
            ),
        )
        comp = [q for q in plan.queries if q.strategy is SearchStrategy.COMPANY_AND_ROLE]
        assert comp[0].text == 'Pfizer "VP AI"'


class TestAtsScoped:
    def test_includes_site_filter(self):
        plan = build_search_plan(
            _profile(industries=["life sciences"]),
            SearchCriteria(target_titles=["VP AI"]),
        )
        ats = [q for q in plan.queries if q.strategy is SearchStrategy.ATS_SCOPED]
        assert ats
        for q in ats:
            assert q.site_filter is not None
            assert q.text.startswith(f"site:{q.site_filter}")

    def test_uses_known_ats_hosts(self):
        plan = build_search_plan(
            _profile(),
            SearchCriteria(target_titles=["VP AI"]),
        )
        ats = [q for q in plan.queries if q.strategy is SearchStrategy.ATS_SCOPED]
        for q in ats:
            assert q.site_filter in ATS_SITE_FILTERS


class TestUrgentHiring:
    def test_emitted_when_prioritize_urgent_true(self):
        plan = build_search_plan(
            _profile(),
            SearchCriteria(target_titles=["VP AI"], prioritize_urgent=True),
        )
        urgent = [q for q in plan.queries if q.strategy is SearchStrategy.URGENT_HIRING]
        assert urgent
        assert urgent[0].text.startswith('"urgently hiring"')

    def test_omitted_when_prioritize_urgent_false(self):
        plan = build_search_plan(
            _profile(),
            SearchCriteria(target_titles=["VP AI"], prioritize_urgent=False),
        )
        assert not [q for q in plan.queries if q.strategy is SearchStrategy.URGENT_HIRING]


class TestExclusions:
    def test_excluded_companies_appended_to_non_ats_queries(self):
        plan = build_search_plan(
            _profile(),
            SearchCriteria(
                target_titles=["VP AI"],
                excluded_companies=["MegaCorp"],
            ),
        )
        for q in plan.queries:
            if q.strategy is SearchStrategy.ATS_SCOPED:
                # ATS-scoped already has site: filter doing the narrowing;
                # we don't want to corrupt the site: query.
                assert '-"MegaCorp"' not in q.text
            else:
                assert '-"MegaCorp"' in q.text


class TestCaps:
    def test_default_caps_are_respected_per_strategy(self):
        # Build a profile with way more inputs than the caps allow.
        profile = _profile(
            target_archetypes=["VP AI " + str(i) for i in range(20)],
            title_equivalents=["Adj " + str(i) for i in range(20)],
            industries=["industry " + str(i) for i in range(20)],
            search_keywords=["skill " + str(i) for i in range(20)],
            ai_data_cloud_experience=[],
        )
        criteria = SearchCriteria(
            target_titles=[],  # use profile fallback
            preferred_companies=["Co " + str(i) for i in range(20)],
        )
        plan = build_search_plan(profile, criteria)

        for strategy, cap in DEFAULT_QUERY_CAPS.items():
            count = sum(1 for q in plan.queries if q.strategy is strategy)
            assert count <= cap, f"{strategy.value} exceeded cap {cap}: got {count}"

    def test_query_caps_override_works(self):
        plan = build_search_plan(
            _profile(),
            SearchCriteria(target_titles=["A", "B", "C", "D"]),
            query_caps={SearchStrategy.EXACT_TITLE: 1},
        )
        exact = [q for q in plan.queries if q.strategy is SearchStrategy.EXACT_TITLE]
        assert len(exact) == 1


class TestDedup:
    def test_case_insensitive_duplicates_collapsed(self):
        profile = _profile(target_archetypes=["VP AI", "vp ai", "VP AI "])
        plan = build_search_plan(profile, SearchCriteria())
        exact = [q for q in plan.queries if q.strategy is SearchStrategy.EXACT_TITLE]
        # Only one unique title survives.
        assert {q.text for q in exact} == {'"VP AI"'}
