"""HTTP liveness check used by the Validation Agent.

A small wrapper around httpx that:
- Follows redirects (so the canonical/final URL is what we hash on).
- Captures the final URL, status, redirect flag, body text, content-type.
- Never raises on transport errors — those are returned as a `FetchResult`
  with `error` set. The Validation Agent decides what to do.

The client is injectable so tests substitute `httpx.MockTransport`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

DEFAULT_USER_AGENT = "ExecutiveJobScout/0.1 (+local-run)"


@dataclass
class FetchResult:
    """Outcome of a single HTTP fetch."""

    requested_url: str
    final_url: Optional[str] = None
    status_code: Optional[int] = None
    redirected: bool = False
    body: str = ""
    content_type: Optional[str] = None
    headers: dict[str, str] | None = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and (self.status_code or 0) // 100 == 2


def fetch_url(
    url: str,
    *,
    client: httpx.Client,
    user_agent: str = DEFAULT_USER_AGENT,
    accept_language: str = "en-US,en;q=0.9",
    timeout_seconds: int = 15,
) -> FetchResult:
    """Fetch a URL and return a FetchResult. Never raises."""
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": accept_language,
    }
    try:
        resp = client.get(
            url,
            headers=headers,
            follow_redirects=True,
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as e:
        return FetchResult(requested_url=url, error=f"transport error: {e}")
    except Exception as e:  # belt and braces - we never want this to bubble
        return FetchResult(requested_url=url, error=f"unexpected error: {e}")

    final_url = str(resp.url)
    return FetchResult(
        requested_url=url,
        final_url=final_url,
        status_code=resp.status_code,
        redirected=final_url != url,
        body=resp.text if resp.status_code // 100 == 2 else "",
        content_type=resp.headers.get("content-type"),
        headers=dict(resp.headers),
    )
