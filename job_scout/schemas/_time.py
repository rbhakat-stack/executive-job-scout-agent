"""Timezone-aware UTC helper.

`datetime.utcnow()` is deprecated in Python 3.12+. All schema defaults route
through `utc_now()` so we get tz-aware UTC datetimes consistently.
"""
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
