"""Reusable Streamlit UI components.

Kept small and stateless: every component takes its data via arguments
and writes to `st.session_state` through the helpers in `src.ui.state`.
"""
from __future__ import annotations

import io
import json
from typing import Optional
from uuid import UUID, uuid4

import pandas as pd
import streamlit as st

from job_scout.config import SESSION_KEY_DISCLOSURE, SESSION_KEY_NAMES
from job_scout.schemas import (
    CandidateProfile,
    JobReport,
    RejectionLogEntry,
    RunRecord,
    UserActions,
)
from job_scout.ui.state import key_store, user_actions_map


# ---------------------------------------------------------------------------
# Settings panel (sidebar)
# ---------------------------------------------------------------------------

def render_settings_panel() -> tuple[str, str]:
    """Render the API-key + provider settings. Returns (provider, model)."""
    st.subheader("Settings")
    st.caption(SESSION_KEY_DISCLOSURE)

    store = key_store()

    provider = st.selectbox(
        "LLM provider",
        options=["anthropic", "openai", "groq", "(none - deterministic only)"],
        index=0,
        help="If '(none)' is selected, the Scoring Agent uses only the "
             "deterministic baseline rationale.",
    )

    model_default = {
        "anthropic": "claude-opus-4-7",
        "openai": "gpt-4o",
        "groq": "llama-3.3-70b-versatile",
        "(none - deterministic only)": "",
    }[provider]
    model = st.text_input("Model", value=model_default, disabled=(provider.startswith("(none")))

    # Per-provider key input (session-only).
    if provider == "anthropic":
        v = st.text_input(
            "ANTHROPIC_API_KEY (session only)",
            type="password",
            value=store.get("ANTHROPIC_API_KEY") or "",
        )
        store.set("ANTHROPIC_API_KEY", v)
    elif provider == "openai":
        v = st.text_input(
            "OPENAI_API_KEY (session only)",
            type="password",
            value=store.get("OPENAI_API_KEY") or "",
        )
        store.set("OPENAI_API_KEY", v)
    elif provider == "groq":
        v = st.text_input(
            "GROQ_API_KEY (session only)",
            type="password",
            value=store.get("GROQ_API_KEY") or "",
        )
        store.set("GROQ_API_KEY", v)

    # Tavily key (always needed for search).
    v = st.text_input(
        "TAVILY_API_KEY (session only)",
        type="password",
        value=store.get("TAVILY_API_KEY") or "",
        help="Tavily is the default search provider.",
    )
    store.set("TAVILY_API_KEY", v)

    return provider, model


# ---------------------------------------------------------------------------
# Profile preview
# ---------------------------------------------------------------------------

def render_profile_preview(profile: CandidateProfile) -> None:
    st.subheader("Extracted profile")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Seniority**")
        st.write(profile.seniority_level.value)
        st.markdown("**Industries**")
        st.write(", ".join(profile.industries) or "_(none extracted)_")
        st.markdown("**Functional expertise**")
        st.write(", ".join(profile.functional_expertise) or "_(none)_")
        st.markdown("**Leadership scope**")
        st.write(profile.leadership_scope or "_(none)_")
    with cols[1]:
        st.markdown("**Target archetypes**")
        st.write(", ".join(profile.target_archetypes) or "_(none)_")
        st.markdown("**Title equivalents**")
        st.write(", ".join(profile.title_equivalents) or "_(none)_")
        st.markdown("**AI / data / cloud experience**")
        st.write(", ".join(profile.ai_data_cloud_experience) or "_(none)_")
        st.markdown("**Search keywords**")
        st.write(", ".join(profile.search_keywords) or "_(none)_")

    with st.expander("Summary"):
        st.write(profile.summary)

    with st.expander("Full JSON"):
        st.code(profile.model_dump_json(indent=2), language="json")


# ---------------------------------------------------------------------------
# Results table + filters
# ---------------------------------------------------------------------------

def _row_for_report(report: JobReport, actions: Optional[UserActions]) -> dict:
    j = report.job
    return {
        "Match": report.score.match_score,
        "Urgency": report.score.urgency_score,
        "Title": j.title,
        "Company": j.company,
        "Location": j.location or "",
        "Work mode": j.work_mode.value,
        "Posted": j.freshness.posted_at.isoformat() if j.freshness.posted_at else "",
        "Freshness": j.freshness.label.value,
        "Source": (j.ats.value if j.ats else "web"),
        "Apply URL": str(j.apply_url),
        "Favorited": bool(actions and actions.favorited),
        "Applied": bool(actions and actions.applied),
        "dedup_hash": j.dedup_hash,
    }


def render_results_filters(run: RunRecord) -> dict:
    """Render filter widgets for the results table. Returns the filter dict."""
    st.subheader("Filters")
    cols = st.columns(5)
    with cols[0]:
        min_match = st.slider("Min match", 0, 100, value=0, step=5)
    with cols[1]:
        min_urgency = st.slider("Min urgency", 0, 100, value=0, step=5)
    with cols[2]:
        companies = sorted({r.job.company for r in run.reports})
        company = st.selectbox("Company", ["(any)"] + companies)
    with cols[3]:
        locations = sorted({r.job.location for r in run.reports if r.job.location})
        location = st.selectbox("Location", ["(any)"] + locations)
    with cols[4]:
        sources = sorted({(r.job.ats.value if r.job.ats else "web") for r in run.reports})
        source = st.selectbox("Source", ["(any)"] + sources)

    return {
        "min_match": min_match,
        "min_urgency": min_urgency,
        "company": company,
        "location": location,
        "source": source,
    }


def _apply_filters(reports: list[JobReport], filters: dict) -> list[JobReport]:
    out = reports
    if filters["min_match"] > 0:
        out = [r for r in out if r.score.match_score >= filters["min_match"]]
    if filters["min_urgency"] > 0:
        out = [r for r in out if r.score.urgency_score >= filters["min_urgency"]]
    if filters["company"] != "(any)":
        out = [r for r in out if r.job.company == filters["company"]]
    if filters["location"] != "(any)":
        out = [r for r in out if r.job.location == filters["location"]]
    if filters["source"] != "(any)":
        out = [
            r
            for r in out
            if (r.job.ats.value if r.job.ats else "web") == filters["source"]
        ]
    return out


def render_results_table(run: RunRecord, filters: dict) -> list[JobReport]:
    """Render the filterable results table. Returns filtered reports."""
    actions = user_actions_map()
    filtered = _apply_filters(run.reports, filters)

    if not filtered:
        st.info("No jobs match the current filters.")
        return []

    rows = [_row_for_report(r, actions.get(r.job.dedup_hash)) for r in filtered]
    df = pd.DataFrame(rows)
    st.dataframe(
        df.drop(columns=["dedup_hash"]),
        width="stretch",
        hide_index=True,
        column_config={
            "Apply URL": st.column_config.LinkColumn(),
            "Match": st.column_config.ProgressColumn(min_value=0, max_value=100),
            "Urgency": st.column_config.ProgressColumn(min_value=0, max_value=100),
        },
    )

    # Export buttons.
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export filtered to CSV",
        data=csv_bytes,
        file_name="executive_job_scout_results.csv",
        mime="text/csv",
    )

    return filtered


# ---------------------------------------------------------------------------
# Detailed job card
# ---------------------------------------------------------------------------

def render_job_card(
    report: JobReport,
    profile_id: UUID,
) -> None:
    j = report.job
    actions_map = user_actions_map()
    actions = actions_map.get(j.dedup_hash) or UserActions(
        profile_id=profile_id, job_id=uuid4()
    )

    title = f"{j.title} - {j.company}"
    with st.expander(f"{title}  -  match {report.score.match_score}/100  -  urgency {report.score.urgency_score}/100"):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown(f"**Location**: {j.location or '_unknown_'}")
            st.markdown(f"**Work mode**: {j.work_mode.value}")
        with c2:
            st.markdown(f"**Posted**: {j.freshness.posted_at or '_unknown_'}")
            st.markdown(f"**Freshness**: {j.freshness.label.value}")
        with c3:
            st.markdown(f"**Source**: {(j.ats.value if j.ats else 'web')}")
            st.link_button("Apply", url=str(j.apply_url))

        st.markdown("---")
        st.markdown("**Match rationale**")
        st.write(report.score.match_rationale)
        if report.score.concerns:
            st.markdown("**Concerns**")
            st.write(report.score.concerns)
        if report.score.application_angle:
            st.markdown("**Application angle**")
            st.write(report.score.application_angle)
        if report.score.outreach_angle:
            st.markdown("**Outreach angle**")
            st.write(report.score.outreach_angle)

        # Citations grouped by claim_key.
        if report.evidence.citations:
            st.markdown("**Evidence**")
            by_key: dict[str, list] = {}
            for c in report.evidence.citations:
                by_key.setdefault(c.claim_key, []).append(c)
            for key, cites in by_key.items():
                with st.container():
                    st.markdown(f"_`{key}`_")
                    for c in cites:
                        st.caption(f"“{c.quote}”")

        # Per-job user actions.
        st.markdown("---")
        a_cols = st.columns([1, 1, 4])
        with a_cols[0]:
            fav = st.checkbox("Favorite", value=actions.favorited, key=f"fav_{j.dedup_hash}")
        with a_cols[1]:
            applied = st.checkbox("Applied", value=actions.applied, key=f"app_{j.dedup_hash}")
        with a_cols[2]:
            notes = st.text_input(
                "Notes",
                value=actions.notes or "",
                key=f"notes_{j.dedup_hash}",
                placeholder="Add a note (session only)",
            )

        # Persist back into the session map.
        actions_map[j.dedup_hash] = UserActions(
            profile_id=profile_id,
            job_id=actions.job_id,
            favorited=fav,
            applied=applied,
            notes=notes or None,
        )


# ---------------------------------------------------------------------------
# Run log / metrics
# ---------------------------------------------------------------------------

def render_run_metrics(run: RunRecord) -> None:
    st.subheader("Run metrics")
    cols = st.columns(4)
    cols[0].metric("Discovered", run.metrics.discovered)
    cols[1].metric("Validated", run.metrics.validated)
    cols[2].metric("Surfaced", run.metrics.surfaced)
    cols[3].metric("Latency (ms)", run.metrics.latency_ms)

    # LLM cost row only shows when a model was actually used.
    if run.metrics.tokens_in or run.metrics.tokens_out:
        cols = st.columns(4)
        cols[0].metric("Tokens in", run.metrics.tokens_in)
        cols[1].metric("Tokens out", run.metrics.tokens_out)
        cols[2].metric("Cost (USD)", f"${run.metrics.cost_usd:.4f}")
        cols[3].metric("Model", run.llm_model or "-")

    if run.rejection_log:
        with st.expander(f"Rejection log ({len(run.rejection_log)})"):
            df = pd.DataFrame(
                [
                    {
                        "stage": r.stage,
                        "url": r.url or "",
                        "reason": r.reason,
                    }
                    for r in run.rejection_log
                ]
            )
            st.dataframe(df, width="stretch", hide_index=True)
