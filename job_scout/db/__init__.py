"""Public DB exports + factory."""
from __future__ import annotations

from typing import Optional

from job_scout.config import Settings, get_settings

from .base import (
    JobRepo,
    ProfileRepo,
    Repositories,
    RunRepo,
    UserActionsRepo,
)
from .memory_repo import InMemoryRepositories
from .supabase_repo import SupabaseRepositories

__all__ = [
    "ProfileRepo",
    "JobRepo",
    "RunRepo",
    "UserActionsRepo",
    "Repositories",
    "InMemoryRepositories",
    "SupabaseRepositories",
    "build_repositories",
]


def build_repositories(settings: Optional[Settings] = None):
    """Pick the repository implementation based on configuration.

    Returns `SupabaseRepositories` if SUPABASE_ENABLED and credentials are
    present; otherwise `InMemoryRepositories`. The orchestrator catches
    Supabase connectivity errors at runtime and falls back to in-memory.
    """
    s = settings or get_settings()
    if s.SUPABASE_ENABLED and s.SUPABASE_URL and s.SUPABASE_ANON_KEY:
        return SupabaseRepositories(s.SUPABASE_URL, s.SUPABASE_ANON_KEY)
    return InMemoryRepositories()
