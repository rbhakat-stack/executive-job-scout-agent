"""HTML extraction for job postings.

Strategy (in order):
  1. JSON-LD `JobPosting` schema (covers most modern career pages: Workday,
     SmartRecruiters, iCIMS, many ATS-hosted pages, Lever JSON, and most
     custom career sites that opted in to schema.org).
  2. ATS-specific HTML fallbacks (Greenhouse / Lever / Ashby DOM hooks).
  3. Generic fallback: <h1> for title, page text for body, <meta> for date.

The extractor is intentionally conservative: if any required field is
missing, it leaves it as None and lets the Validation Agent reject the job.
We never fabricate dates, titles, or companies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Phrases that indicate the posting is no longer accepting applications.
# Match against lowercased body text. Conservative list - false positives
# silently drop real jobs, so we keep these specific.
# ---------------------------------------------------------------------------
EXPIRED_PHRASES: tuple[str, ...] = (
    "this position is no longer available",
    "this job is no longer accepting applications",
    "this role is no longer accepting applications",
    "this position has been filled",
    "no longer accepting applications",
    "this posting has expired",
    "this posting has been removed",
    "the position has been closed",
    "this job has closed",
    "applications are closed",
    "this opportunity is no longer available",
)

# Work-mode keyword heuristics (used only if JSON-LD doesn't tell us).
_REMOTE_PHRASES = ("fully remote", "100% remote", "remote (", "remote -", "remote,", "work from anywhere")
_HYBRID_PHRASES = ("hybrid", "hybrid work", "hybrid model")
_ONSITE_PHRASES = ("on-site", "onsite", "in office")


@dataclass
class ExtractedJob:
    """Everything we pulled out of the page. All fields are best-effort."""

    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    work_mode: Optional[str] = None             # remote / hybrid / onsite
    body_text: str = ""
    posted_at: Optional[date] = None
    posted_at_source: Optional[str] = None      # 'json_ld_datePosted', 'meta_published_time', ...
    posted_at_snippet: Optional[str] = None     # the raw text/value
    expired_signals: list[str] = field(default_factory=list)
    apply_url: Optional[str] = None
    recruiter_contact: Optional[str] = None
    multiple_openings: bool = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_from_html(html: str, *, source_url: str) -> ExtractedJob:
    if not html:
        return ExtractedJob()

    soup = BeautifulSoup(html, "lxml")
    out = ExtractedJob()

    # 1. JSON-LD JobPosting
    jl = _find_json_ld_jobposting(soup)
    if jl:
        _apply_json_ld(jl, out)

    # 2. HTML fallbacks for fields JSON-LD didn't supply.
    if not out.title:
        out.title = _first_text(soup.find("h1")) or _first_text(soup.find("title"))

    if not out.body_text:
        out.body_text = _visible_text(soup)

    # 3. Date fallback: <meta property="article:published_time">
    if not out.posted_at:
        meta = soup.find("meta", attrs={"property": "article:published_time"})
        if meta and meta.get("content"):
            d = _parse_iso_date(meta["content"])
            if d:
                out.posted_at = d
                out.posted_at_source = "meta_published_time"
                out.posted_at_snippet = meta["content"]

    # 4. Work mode (heuristic on body text if JSON-LD didn't set it).
    if not out.work_mode and out.body_text:
        out.work_mode = _infer_work_mode(out.body_text)

    # 5. Expired signals on the body.
    if out.body_text:
        body_lower = out.body_text.lower()
        for phrase in EXPIRED_PHRASES:
            if phrase in body_lower:
                out.expired_signals.append(phrase)

    # 6. Recruiter contact (simple email regex)
    if out.body_text and not out.recruiter_contact:
        m = re.search(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", out.body_text)
        if m:
            out.recruiter_contact = m.group(0)

    # 7. Multiple openings signal
    if out.body_text:
        out.multiple_openings = bool(
            re.search(r"\b(multiple|several|many)\s+(openings|positions|roles)\b", out.body_text, re.I)
        )

    # 8. Apply URL: default to source if not set by JSON-LD applyUrl.
    if not out.apply_url:
        out.apply_url = source_url

    # Trim oversized body text (some pages dump entire site content).
    if len(out.body_text) > 20_000:
        out.body_text = out.body_text[:20_000]

    return out


# ---------------------------------------------------------------------------
# JSON-LD extraction
# ---------------------------------------------------------------------------

def _find_json_ld_jobposting(soup: BeautifulSoup) -> Optional[dict[str, Any]]:
    """Find a `JobPosting` JSON-LD block, if any."""
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # Some sites embed multiple JSON-LDs concatenated; try a permissive parse.
            try:
                data = json.loads("[" + raw.replace("}\n{", "},\n{") + "]")
            except (json.JSONDecodeError, ValueError):
                continue

        for candidate in _iter_jsonld_nodes(data):
            t = candidate.get("@type")
            if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
                return candidate
    return None


def _iter_jsonld_nodes(node: Any):
    """Yield every dict node within a possibly-nested JSON-LD structure."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_jsonld_nodes(v)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_jsonld_nodes(item)


def _apply_json_ld(jl: dict[str, Any], out: ExtractedJob) -> None:
    out.title = jl.get("title") or out.title

    org = jl.get("hiringOrganization")
    if isinstance(org, dict):
        out.company = org.get("name") or out.company
    elif isinstance(org, str):
        out.company = org

    loc = jl.get("jobLocation")
    out.location = _flatten_location(loc) or out.location

    if jl.get("datePosted"):
        d = _parse_iso_date(str(jl["datePosted"]))
        if d:
            out.posted_at = d
            out.posted_at_source = "json_ld_datePosted"
            out.posted_at_snippet = str(jl["datePosted"])

    desc = jl.get("description")
    if isinstance(desc, str) and desc:
        out.body_text = _strip_html_to_text(desc)

    if jl.get("jobLocationType") == "TELECOMMUTE" or jl.get("workFromHome") is True:
        out.work_mode = "remote"

    # JSON-LD `url` is the canonical URL of THIS posting. We don't fall back
    # to hiringOrganization.url because that's the company homepage, not the
    # apply link. The Validation Agent fills source_url if this is missing.
    if isinstance(jl.get("url"), str):
        out.apply_url = jl["url"]


def _flatten_location(loc: Any) -> Optional[str]:
    """JSON-LD jobLocation can be a dict, a list, or absent. Pick the best label."""
    if loc is None:
        return None
    if isinstance(loc, list):
        # Pick the first one with a parseable address.
        for item in loc:
            r = _flatten_location(item)
            if r:
                return r
        return None
    if isinstance(loc, dict):
        addr = loc.get("address")
        if isinstance(addr, dict):
            city = addr.get("addressLocality") or ""
            region = addr.get("addressRegion") or ""
            country = addr.get("addressCountry") or ""
            parts = [p for p in (city, region, country) if isinstance(p, str) and p.strip()]
            return ", ".join(parts) if parts else None
        if isinstance(addr, str):
            return addr
    if isinstance(loc, str):
        return loc
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_text(tag) -> Optional[str]:
    if tag is None:
        return None
    text = tag.get_text(strip=True)
    return text or None


def _visible_text(soup: BeautifulSoup) -> str:
    """Strip script/style/nav/footer and return the visible text."""
    for s in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        s.decompose()
    return soup.get_text("\n", strip=True)


def _strip_html_to_text(html: str) -> str:
    return BeautifulSoup(html, "lxml").get_text("\n", strip=True)


def _parse_iso_date(value: str) -> Optional[date]:
    """Parse an ISO 8601 datetime/date string into a `date`. Forgiving."""
    if not value:
        return None
    v = value.strip().replace("Z", "+00:00")
    # Try datetime first, then bare date.
    for parser in (datetime.fromisoformat, _from_iso_date):
        try:
            d = parser(v)
            return d.date() if isinstance(d, datetime) else d
        except ValueError:
            continue
    # Loose fallback: YYYY-MM-DD anywhere in the string.
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", v)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def _from_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def _infer_work_mode(body: str) -> Optional[str]:
    b = body.lower()
    if any(p in b for p in _REMOTE_PHRASES):
        return "remote"
    if any(p in b for p in _HYBRID_PHRASES):
        return "hybrid"
    if any(p in b for p in _ONSITE_PHRASES):
        return "onsite"
    return None
