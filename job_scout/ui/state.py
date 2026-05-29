"""Streamlit session-state plumbing.

Initializes a `SessionKeyStore` backed by `st.session_state`, plus typed
slots for the profile, last run, and per-job user actions (favorites,
applied, notes). API keys live ONLY in `st.session_state` — they are
never written to disk, the database, or any log.
"""
from __future__ import annotations

from typing import Optional
from uuid import uuid4

import streamlit as st

from job_scout.config import SessionKeyStore
from job_scout.schemas import CandidateProfile, RunRecord, UserActions

# Top-level session_state keys.
KEY_STORE = "__api_keys"
KEY_PROFILE = "__profile"
KEY_PROFILE_ID = "__profile_id"
KEY_LAST_RUN = "__last_run"
KEY_USER_ACTIONS = "__user_actions"  # dict[dedup_hash, UserActions]


def init_session_state() -> None:
    """Idempotently ensure the session-state slots exist.

    Streamlit reruns the script on every interaction, so this is called at
    the top of `app.py` to make sure mutations don't bounce off missing keys.
    """
    if KEY_STORE not in st.session_state:
        st.session_state[KEY_STORE] = {}
    if KEY_PROFILE not in st.session_state:
        st.session_state[KEY_PROFILE] = None
    if KEY_PROFILE_ID not in st.session_state:
        st.session_state[KEY_PROFILE_ID] = uuid4()
    if KEY_LAST_RUN not in st.session_state:
        st.session_state[KEY_LAST_RUN] = None
    if KEY_USER_ACTIONS not in st.session_state:
        st.session_state[KEY_USER_ACTIONS] = {}


def key_store() -> SessionKeyStore:
    """Return a `SessionKeyStore` backed by the active Streamlit session."""
    return SessionKeyStore(backing=st.session_state[KEY_STORE])


def get_profile() -> Optional[CandidateProfile]:
    return st.session_state.get(KEY_PROFILE)


def set_profile(profile: CandidateProfile) -> None:
    st.session_state[KEY_PROFILE] = profile


def clear_profile() -> None:
    st.session_state[KEY_PROFILE] = None


def get_last_run() -> Optional[RunRecord]:
    return st.session_state.get(KEY_LAST_RUN)


def set_last_run(run: RunRecord) -> None:
    st.session_state[KEY_LAST_RUN] = run


def user_actions_map() -> dict[str, UserActions]:
    """Mutable per-session dict of `dedup_hash -> UserActions`.

    Held in memory only; the Supabase `user_actions` repo is wired in
    M9 for persistence across sessions.
    """
    return st.session_state[KEY_USER_ACTIONS]
