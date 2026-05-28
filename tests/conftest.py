"""Shared pytest fixtures.

Concrete fixtures are added by the milestones that need them (M2 onward).
This file exists so pytest discovery and rootdir resolution are stable from M0.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `import src.*` works in tests
# regardless of how pytest is invoked.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
