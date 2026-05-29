"""Tests for the session-only key store and `resolve_secret()` precedence.

The non-negotiable here: a session value MUST override the env, and the
store MUST refuse keys outside the SESSION_KEY_NAMES allowlist (defense in
depth so we don't accidentally start persisting other secrets).
"""
from __future__ import annotations

import pytest

from job_scout.config import (
    SESSION_KEY_NAMES,
    SessionKeyStore,
    Settings,
    resolve_secret,
)


class TestSessionKeyStore:
    def test_set_and_get_allowed_key(self):
        s = SessionKeyStore()
        s.set("ANTHROPIC_API_KEY", "sk-test")
        assert s.get("ANTHROPIC_API_KEY") == "sk-test"
        assert "ANTHROPIC_API_KEY" in s

    def test_set_unknown_key_is_rejected(self):
        s = SessionKeyStore()
        with pytest.raises(ValueError, match="unknown key"):
            s.set("SUPABASE_SERVICE_ROLE_KEY", "should-not-store")

    def test_empty_value_clears_the_key(self):
        s = SessionKeyStore()
        s.set("TAVILY_API_KEY", "tvly-x")
        s.set("TAVILY_API_KEY", "")
        assert s.get("TAVILY_API_KEY") is None

    def test_whitespace_is_stripped(self):
        s = SessionKeyStore()
        s.set("OPENAI_API_KEY", "  sk-pad  ")
        assert s.get("OPENAI_API_KEY") == "sk-pad"

    def test_clear_only_touches_session_keys(self):
        backing = {
            "ANTHROPIC_API_KEY": "session-val",
            "UNRELATED": "leave-me-alone",
        }
        s = SessionKeyStore(backing=backing)
        s.clear()
        assert "ANTHROPIC_API_KEY" not in backing
        assert backing["UNRELATED"] == "leave-me-alone"


class TestResolveSecret:
    def test_session_overrides_env(self):
        settings = Settings(ANTHROPIC_API_KEY="from-env")
        session = SessionKeyStore()
        session.set("ANTHROPIC_API_KEY", "from-session")
        assert (
            resolve_secret("ANTHROPIC_API_KEY", session=session, settings=settings)
            == "from-session"
        )

    def test_falls_back_to_env_when_session_missing(self):
        settings = Settings(ANTHROPIC_API_KEY="from-env")
        session = SessionKeyStore()
        assert (
            resolve_secret("ANTHROPIC_API_KEY", session=session, settings=settings)
            == "from-env"
        )

    def test_returns_none_when_neither_set(self):
        settings = Settings(ANTHROPIC_API_KEY=None)
        session = SessionKeyStore()
        assert (
            resolve_secret("ANTHROPIC_API_KEY", session=session, settings=settings)
            is None
        )

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError):
            resolve_secret("SUPABASE_SERVICE_ROLE_KEY")

    def test_all_session_key_names_are_settings_fields(self):
        s = Settings()
        for name in SESSION_KEY_NAMES:
            assert hasattr(s, name), f"Settings is missing {name}"
