"""Configuration and the session-only API key store.

Two layers:

1. `Settings` (pydantic-settings) — loaded once from environment variables and
   optional `.env`. Holds non-secret defaults (provider names, timeouts, model
   ids, freshness defaults) and the *fallback* values for API keys when the
   user has not entered any in the UI.

2. `SessionKeyStore` — runtime-only key holder, scoped to a single Streamlit
   session via `st.session_state`. Keys entered in the UI Settings page live
   here ONLY. They are never written to the database or to disk.

`resolve_secret()` is the canonical accessor: it returns the session value
if present, else the env value, else None. Agents must use this helper —
never read API keys directly from `os.environ` or `Settings`.
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Keys exposed via the UI Settings page (session-only override) ---
SESSION_KEY_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
)

SESSION_KEY_DISCLOSURE = (
    "Your key is only visible and accessible during this logged-in session "
    "and is not stored in the database."
)


class Settings(BaseSettings):
    """Environment-derived defaults. Never holds runtime user-entered keys."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- LLM ---
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = "claude-opus-4-7"
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # --- Search ---
    SEARCH_PROVIDER: str = "tavily"
    TAVILY_API_KEY: Optional[str] = None

    # --- Supabase ---
    SUPABASE_ENABLED: bool = True
    SUPABASE_URL: Optional[str] = None
    SUPABASE_ANON_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None

    # --- App ---
    APP_ENV: str = "local"
    LOG_LEVEL: str = "INFO"
    DEFAULT_POSTING_MAX_AGE_DAYS: int = Field(default=45, ge=1, le=365)
    HTTP_TIMEOUT_SECONDS: int = 15
    HTTP_USER_AGENT: str = "ExecutiveJobScout/0.1 (+local-run)"


_settings_singleton: Optional[Settings] = None


def get_settings() -> Settings:
    """Lazily construct a process-wide `Settings`. Cheap; safe to call often."""
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = Settings()
    return _settings_singleton


# ---------------------------------------------------------------------------
# Session-only key store
# ---------------------------------------------------------------------------

class SessionKeyStore:
    """A dict-like holder for session-scoped API keys.

    Designed to wrap `st.session_state` (which is itself dict-like) but works
    standalone for tests. Keys live ONLY here for the lifetime of the session.
    """

    def __init__(self, backing: Optional[dict] = None) -> None:
        self._backing: dict = backing if backing is not None else {}

    def set(self, name: str, value: Optional[str]) -> None:
        if name not in SESSION_KEY_NAMES:
            raise ValueError(
                f"Refusing to store unknown key '{name}'. "
                f"Allowed: {SESSION_KEY_NAMES}"
            )
        # Never store empty strings; treat as "user cleared the field".
        if value is None or value.strip() == "":
            self._backing.pop(name, None)
            return
        self._backing[name] = value.strip()

    def get(self, name: str) -> Optional[str]:
        v = self._backing.get(name)
        return v if v else None

    def clear(self) -> None:
        for k in list(self._backing.keys()):
            if k in SESSION_KEY_NAMES:
                self._backing.pop(k, None)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None


def resolve_secret(
    name: str,
    session: Optional[SessionKeyStore] = None,
    settings: Optional[Settings] = None,
) -> Optional[str]:
    """Return the effective value for an API key.

    Precedence: session > env (`Settings`) > None. Agents MUST use this.
    """
    if name not in SESSION_KEY_NAMES:
        raise ValueError(f"resolve_secret: unknown key '{name}'.")
    if session is not None:
        v = session.get(name)
        if v:
            return v
    s = settings or get_settings()
    return getattr(s, name, None)
