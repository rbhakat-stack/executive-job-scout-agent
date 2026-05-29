"""Streamlit page sections.

The app is single-page; sections are rendered top-to-bottom under tabs.
State is held in `st.session_state` (see `src/ui/state.py`). All API keys
are session-only.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx
import streamlit as st

from job_scout.agents.profile import ProfileAgent, ProfileAgentError
from job_scout.config import get_settings, resolve_secret
from job_scout.db import SupabaseRepositories
from job_scout.llm import build_llm, FakeLLM, LLM, LLMError
from job_scout.orchestrator import Orchestrator
from job_scout.schemas import (
    CandidateProfile,
    RunRecord,
    SearchCriteria,
    SeniorityLevel,
    WorkMode,
)
from job_scout.search_providers import TavilySearchProvider
from job_scout.search_providers.base import SearchProviderError

# In-package imports use relative form. Python 3.14's import system is
# stricter about partially-initialized parent packages; an absolute
# `from job_scout.ui.components import ...` inside src/ui/pages.py can trip a
# `KeyError: 'src.ui'` during the very first load of the package tree
# (which is exactly when app.py first imports this module).
from .components import (
    render_job_card,
    render_profile_preview,
    render_results_filters,
    render_results_table,
    render_run_metrics,
    render_settings_panel,
)
from .state import (
    KEY_PROFILE_ID,
    clear_profile,
    get_last_run,
    get_profile,
    init_session_state,
    key_store,
    set_last_run,
    set_profile,
)

PROFILE_LLM_REQUIRED_NOTE = (
    "Profile extraction requires an LLM. Choose Anthropic or OpenAI in the "
    "Settings panel and supply an API key (session-only)."
)


# ---------------------------------------------------------------------------
# Section: profile upload
# ---------------------------------------------------------------------------

def render_profile_section(provider: str, model: str) -> None:
    st.header("1. Resume + LinkedIn")
    st.caption("Upload a resume (PDF, DOCX, TXT). Optionally paste a LinkedIn URL or profile text.")

    uploaded = st.file_uploader(
        "Resume", type=["pdf", "docx", "txt"], accept_multiple_files=False
    )
    linkedin_url = st.text_input("LinkedIn URL (optional)")
    linkedin_text = st.text_area("LinkedIn profile text (optional, pasted)", height=120)

    extract_clicked = st.button(
        "Extract profile",
        disabled=(uploaded is None or provider.startswith("(none")),
        type="primary",
    )
    if provider.startswith("(none") and uploaded is not None:
        st.info(PROFILE_LLM_REQUIRED_NOTE)

    if extract_clicked and uploaded is not None:
        llm = _build_llm_or_warn(provider, model)
        if llm is None:
            return
        try:
            with st.spinner("Extracting profile..."):
                agent = ProfileAgent(llm)
                profile = agent.extract(
                    resume_filename=uploaded.name,
                    resume_bytes=uploaded.getvalue(),
                    linkedin_url=linkedin_url or None,
                    linkedin_text=linkedin_text or None,
                )
                set_profile(profile)
            st.success("Profile extracted.")
        except ProfileAgentError as e:
            st.error(f"Profile extraction failed: {e}")
        except LLMError as e:
            # Don't let a transient LLM failure crash the Streamlit script —
            # Streamlit redacts the message in production and the user is
            # left without a clue. Show the actual cause inline.
            st.error(
                f"LLM call failed ({provider}, model='{model}').\n\n"
                f"Details: {e}\n\n"
                "Common causes:\n"
                "- Rate limit (Groq free tier: 30 requests/min). Wait 60s and retry.\n"
                "- Invalid or expired API key — re-enter it in the Settings sidebar.\n"
                "- Model name typo or deprecated model — try 'llama-3.3-70b-versatile' "
                "for Groq or 'gpt-4o' for OpenAI.\n"
                "- Insufficient credits / quota exhausted (check your provider dashboard)."
            )

    profile = get_profile()
    if profile is not None:
        render_profile_preview(profile)
        if st.button("Clear profile"):
            clear_profile()
            st.rerun()


# ---------------------------------------------------------------------------
# Section: criteria + run
# ---------------------------------------------------------------------------

def render_criteria_and_run_section(provider: str, model: str) -> None:
    st.header("2. Search criteria")
    profile = get_profile()
    if profile is None:
        st.info("Extract a profile first to enable search.")
        return

    cols = st.columns(2)
    with cols[0]:
        target_titles_raw = st.text_area(
            "Target titles (one per line)",
            value="\n".join(profile.target_archetypes),
        )
        target_titles = [t.strip() for t in target_titles_raw.splitlines() if t.strip()]

        industries_raw = st.text_area(
            "Preferred industries (one per line)",
            value="\n".join(profile.industries),
        )
        preferred_industries = [t.strip() for t in industries_raw.splitlines() if t.strip()]

        companies_raw = st.text_area(
            "Preferred companies (one per line)",
            value="",
        )
        preferred_companies = [t.strip() for t in companies_raw.splitlines() if t.strip()]

        excluded_raw = st.text_area("Excluded companies (one per line)", value="")
        excluded_companies = [t.strip() for t in excluded_raw.splitlines() if t.strip()]

    with cols[1]:
        location_preference = st.text_input("Location preference (optional)")
        work_modes_labels = st.multiselect(
            "Work modes",
            options=[m.value for m in (WorkMode.REMOTE, WorkMode.HYBRID, WorkMode.ONSITE)],
            default=[WorkMode.REMOTE.value, WorkMode.HYBRID.value, WorkMode.ONSITE.value],
        )
        work_modes = [WorkMode(v) for v in work_modes_labels]
        travel_tolerance = st.text_input("Travel tolerance (free-form, optional)")

        comp_cols = st.columns(2)
        with comp_cols[0]:
            comp_min = st.number_input("Comp min (USD)", min_value=0, value=0, step=10_000)
        with comp_cols[1]:
            comp_max = st.number_input("Comp max (USD)", min_value=0, value=0, step=10_000)

        must_have_raw = st.text_input("Must-have keywords (comma-separated)")
        must_have_keywords = [k.strip() for k in must_have_raw.split(",") if k.strip()]

        exclusion_raw = st.text_input("Exclusion keywords (comma-separated)")
        exclusion_keywords = [k.strip() for k in exclusion_raw.split(",") if k.strip()]

        max_age_days = st.slider(
            "Max posting age (days)",
            min_value=1,
            max_value=180,
            value=45,
            help="Executive search cycles often run 30-90 days. Default 45.",
        )
        allow_older = st.checkbox(
            "Allow older postings (skip the freshness rejection)", value=False
        )
        prioritize_urgent = st.checkbox("Prioritize urgent postings", value=True)
        min_match_score = st.slider(
            "Minimum match score to surface",
            min_value=0,
            max_value=100,
            value=35,
            step=5,
            help="Real Tavily-discovered roles often score 30-50. Tighten if too noisy.",
        )

    try:
        criteria = SearchCriteria(
            target_titles=target_titles,
            preferred_industries=preferred_industries,
            preferred_companies=preferred_companies,
            excluded_companies=excluded_companies,
            location_preference=location_preference or None,
            work_modes=work_modes,
            travel_tolerance=travel_tolerance or None,
            comp_min_usd=comp_min or None,
            comp_max_usd=comp_max or None,
            must_have_keywords=must_have_keywords,
            exclusion_keywords=exclusion_keywords,
            max_age_days=max_age_days,
            allow_older=allow_older,
            prioritize_urgent=prioritize_urgent,
            min_match_score=min_match_score,
        )
    except Exception as e:
        st.error(f"Invalid criteria: {e}")
        return

    st.markdown("---")
    st.header("3. Run search")
    run_clicked = st.button("Run search", type="primary")
    if not run_clicked:
        return

    tavily_key = resolve_secret("TAVILY_API_KEY", session=key_store())
    if not tavily_key:
        st.error("Tavily API key required. Add it in the Settings sidebar.")
        return

    try:
        search_provider = TavilySearchProvider(api_key=tavily_key)
    except SearchProviderError as e:
        st.error(f"Search provider could not initialize: {e}")
        return

    llm = _build_llm_or_warn(provider, model, required=False)

    http_client = httpx.Client(timeout=get_settings().HTTP_TIMEOUT_SECONDS)
    try:
        with st.spinner("Searching, validating, scoring..."):
            orch = Orchestrator(
                search_provider=search_provider,
                http_client=http_client,
                llm=llm,
                llm_model=model if llm else None,
                clock=lambda: datetime.now(timezone.utc).date(),
                user_agent=get_settings().HTTP_USER_AGENT,
                http_timeout_seconds=get_settings().HTTP_TIMEOUT_SECONDS,
            )
            run = orch.run(profile, criteria)
            set_last_run(run)
        st.success(
            f"Run complete: {run.metrics.surfaced} surfaced of {run.metrics.discovered} discovered."
        )
        _persist_run_if_configured(run, profile)
    finally:
        http_client.close()


def _persist_run_if_configured(run: RunRecord, profile: CandidateProfile) -> None:
    """Write the run + jobs + reports to Supabase if configured. Fail-soft.

    Skips silently when SUPABASE_URL/SUPABASE_ANON_KEY are absent, so local
    runs work without cloud setup. On error, shows a UI warning but keeps
    the in-session run intact.
    """
    s = get_settings()
    if not (s.SUPABASE_ENABLED and s.SUPABASE_URL and s.SUPABASE_ANON_KEY):
        st.caption(
            "Supabase persistence disabled (no URL/key configured). "
            "Run held in this session only."
        )
        return

    try:
        repos = SupabaseRepositories(s.SUPABASE_URL, s.SUPABASE_ANON_KEY)
        profile_id = repos.profiles.upsert(profile)
        # Attach profile_id to the run record before insertion.
        run_with_profile = run.model_copy(update={"profile_id": profile_id})
        repos.runs.insert(run_with_profile)
        for report in run.reports:
            job_id = repos.jobs.upsert(report.job)
            repos.runs.attach_report(run.id, job_id, report)
        st.toast("Run persisted to Supabase.")
    except Exception as e:
        st.warning(
            f"Supabase persistence failed (run is still available in this "
            f"session): {e}"
        )


# ---------------------------------------------------------------------------
# Section: results
# ---------------------------------------------------------------------------

def render_results_section() -> None:
    run = get_last_run()
    if run is None:
        st.info("Run a search to see results.")
        return

    render_run_metrics(run)

    if not run.reports:
        st.warning(
            "No jobs surfaced. See the rejection log above for why each lead was filtered out."
        )
        return

    filters = render_results_filters(run)
    filtered = render_results_table(run, filters)

    if filtered:
        st.subheader("Detailed job cards")
        profile_id = st.session_state[KEY_PROFILE_ID]
        for report in filtered:
            render_job_card(report, profile_id=profile_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_llm_or_warn(provider: str, model: str, *, required: bool = True) -> Optional[LLM]:
    if provider.startswith("(none"):
        if required:
            st.error(PROFILE_LLM_REQUIRED_NOTE)
            return None
        return None
    try:
        llm = build_llm(
            provider=provider,
            session=key_store(),
            model=model or None,
        )
    except LLMError as e:
        st.error(str(e))
        return None
    if llm is None:
        if required:
            st.error(
                f"No API key found for provider {provider!r}. "
                f"Enter it in the Settings sidebar (session only)."
            )
        return None
    return llm


# ---------------------------------------------------------------------------
# Top-level layout
# ---------------------------------------------------------------------------

def render_app() -> None:
    init_session_state()
    st.set_page_config(
        page_title="Executive Job Scout Agent",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        provider, model = render_settings_panel()

    st.title("Executive Job Scout Agent")
    st.caption(
        "Find real, recently posted, validated executive roles aligned to "
        "your CV and LinkedIn profile."
    )

    tab_setup, tab_results = st.tabs(["Setup & Run", "Results"])
    with tab_setup:
        render_profile_section(provider, model)
        st.markdown("---")
        render_criteria_and_run_section(provider, model)
    with tab_results:
        render_results_section()
