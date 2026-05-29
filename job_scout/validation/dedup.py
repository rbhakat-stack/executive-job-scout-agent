"""Deduplication hashing.

Two jobs are considered the same posting if they share the same canonical
URL + normalized title + normalized company. Tracking parameters and
fragments are stripped so that `…/jobs/1?utm_source=x` and `…/jobs/1`
collapse to the same hash.

The hash is used:
  * by `JobRepo.upsert` to skip duplicate inserts
  * by the Red Team Agent to reject a second copy of an already-accepted job
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query parameter names that never identify the job — strip them.
_TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_name",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "ref_source",
        "source",
        "src",
        "hsCtaTracking",
        "hss_channel",
    }
)


def canonicalize_url(url: str) -> str:
    """Return a deterministic URL form for dedup comparison.

    Steps: lowercase scheme/host, drop default ports, drop fragment, strip
    tracking params, sort remaining params, strip trailing slash on non-root
    paths.
    """
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()

    # Host: lowercase, drop default port.
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not (
        (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    ):
        host = f"{host}:{port}"

    # Path: strip trailing slash unless the entire path IS "/".
    path = parts.path or ""
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Query: drop tracking params, sort the rest.
    kept_params = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    kept_params.sort()
    query = urlencode(kept_params, doseq=True)

    # Fragment dropped entirely.
    return urlunsplit((scheme, host, path, query, ""))


def _normalize_label(s: str) -> str:
    """Lowercase + collapse whitespace; preserve punctuation that matters
    (hyphens in company names like Bristol-Myers Squibb).
    """
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def compute_dedup_hash(canonical_url: str, title: str, company: str) -> str:
    """Return a stable sha256 hex digest identifying a unique posting."""
    payload = "|".join(
        [
            canonicalize_url(canonical_url),
            _normalize_label(title),
            _normalize_label(company),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
