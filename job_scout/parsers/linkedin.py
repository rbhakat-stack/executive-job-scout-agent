"""LinkedIn profile parsing.

LinkedIn does not permit direct scraping of authenticated profiles, so we
accept either:
  * a profile URL (we just stash it for provenance; we do NOT fetch it), or
  * pasted profile text (the user copies the visible profile content).

This module normalizes pasted text into a `{canonical_section: text}` dict
that the Profile Agent prompt can format compactly.
"""
from __future__ import annotations

import re
from typing import Optional

# Common section headings LinkedIn renders, in the order users see them.
_SECTION_PATTERNS: tuple[str, ...] = (
    "About",
    "Experience",
    "Education",
    "Skills",
    "Licenses & certifications",
    "Licenses and certifications",
    "Certifications",
    "Publications",
    "Honors & awards",
    "Honors and awards",
    "Awards",
    "Languages",
    "Volunteer experience",
    "Projects",
    "Recommendations",
    "Activity",
    "Featured",
)

_HEADER_RE = re.compile(
    r"^(?P<header>" + "|".join(re.escape(s) for s in _SECTION_PATTERNS) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_linkedin_text(raw: Optional[str]) -> dict[str, str]:
    """Split pasted LinkedIn profile text into named sections.

    Returns a dict keyed by a canonical section name (lowercased,
    `&` -> `and`, whitespace -> `_`). Content before the first detected
    header is stored under `__preamble__` (typically headline + summary).
    Empty / falsy input returns an empty dict.
    """
    text = (raw or "").strip()
    if not text:
        return {}

    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return {"__preamble__": text}

    sections: dict[str, str] = {}
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["__preamble__"] = preamble

    for i, m in enumerate(matches):
        name = _canonical(m.group("header"))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            # If the same section appears twice (rare), append.
            sections[name] = (sections.get(name, "") + "\n" + body).strip()

    return sections


def _canonical(header: str) -> str:
    """Normalize 'Licenses & certifications' -> 'licenses_and_certifications'."""
    h = header.strip().lower()
    h = h.replace("&", "and")
    h = re.sub(r"\s+", "_", h)
    h = re.sub(r"[^a-z_]+", "", h)
    return h
