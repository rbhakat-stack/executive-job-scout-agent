"""HTML extractor tests.

Covers:
- JSON-LD `JobPosting` extraction (the modern-page happy path)
- HTML fallbacks when JSON-LD is missing
- Expired-signal detection
- Edge cases: empty body, broken JSON-LD, multiple JSON-LD blocks
"""
from __future__ import annotations

import json

from src.parsers.ats import EXPIRED_PHRASES, extract_from_html


def _html_with_jsonld(payload: dict, body_html: str = "<p>Generic body text.</p>") -> str:
    return (
        "<html><head>"
        f'<script type="application/ld+json">{json.dumps(payload)}</script>'
        "</head>"
        f"<body>{body_html}</body></html>"
    )


JSONLD_OK = {
    "@context": "https://schema.org",
    "@type": "JobPosting",
    "title": "VP AI Transformation",
    "datePosted": "2026-05-20",
    "hiringOrganization": {"@type": "Organization", "name": "Acme Bio"},
    "jobLocation": {
        "@type": "Place",
        "address": {
            "@type": "PostalAddress",
            "addressLocality": "Boston",
            "addressRegion": "MA",
            "addressCountry": "US",
        },
    },
    "jobLocationType": "TELECOMMUTE",
    "description": "<p>Lead AI transformation across pharma.</p>",
}


class TestJsonLd:
    def test_happy_path(self):
        out = extract_from_html(
            _html_with_jsonld(JSONLD_OK),
            source_url="https://boards.greenhouse.io/acme/jobs/1",
        )
        assert out.title == "VP AI Transformation"
        assert out.company == "Acme Bio"
        assert "Boston" in (out.location or "")
        assert out.work_mode == "remote"
        assert out.posted_at is not None
        assert out.posted_at.isoformat() == "2026-05-20"
        assert out.posted_at_source == "json_ld_datePosted"
        assert "Lead AI transformation" in out.body_text

    def test_string_hiring_org(self):
        p = dict(JSONLD_OK)
        p["hiringOrganization"] = "Acme Bio"
        out = extract_from_html(_html_with_jsonld(p), source_url="https://x/jobs/1")
        assert out.company == "Acme Bio"

    def test_iso_datetime_z_suffix(self):
        p = dict(JSONLD_OK)
        p["datePosted"] = "2026-05-20T00:00:00Z"
        out = extract_from_html(_html_with_jsonld(p), source_url="https://x/jobs/1")
        assert out.posted_at.isoformat() == "2026-05-20"

    def test_jsonld_in_a_list(self):
        p = [JSONLD_OK, {"@type": "BreadcrumbList", "itemListElement": []}]
        html = (
            "<html><head>"
            f'<script type="application/ld+json">{json.dumps(p)}</script>'
            "</head><body></body></html>"
        )
        out = extract_from_html(html, source_url="https://x/jobs/1")
        assert out.title == "VP AI Transformation"

    def test_broken_jsonld_does_not_explode(self):
        html = (
            '<html><head>'
            '<script type="application/ld+json">{not valid json</script>'
            '</head><body><h1>Fallback Title</h1>'
            '<p>Body content that is long enough to be substantive.</p></body></html>'
        )
        out = extract_from_html(html, source_url="https://x/jobs/1")
        # Falls back to <h1> for title.
        assert out.title == "Fallback Title"

    def test_apply_url_falls_back_to_source(self):
        # JSON-LD provides no 'url' field; the extractor should fall back.
        p = dict(JSONLD_OK)
        p.pop("url", None)
        out = extract_from_html(
            _html_with_jsonld(p),
            source_url="https://boards.greenhouse.io/acme/jobs/1",
        )
        assert out.apply_url == "https://boards.greenhouse.io/acme/jobs/1"


class TestHtmlFallbacks:
    def test_title_from_h1(self):
        html = (
            "<html><body>"
            "<h1>Head of AI</h1>"
            "<p>Body paragraph one.</p>"
            "<p>Body paragraph two.</p>"
            "</body></html>"
        )
        out = extract_from_html(html, source_url="https://x/jobs/1")
        assert out.title == "Head of AI"
        assert "Body paragraph" in out.body_text

    def test_meta_published_time_used_when_no_jsonld_date(self):
        html = (
            '<html><head>'
            '<meta property="article:published_time" content="2026-05-21T00:00:00Z">'
            '</head><body><h1>VP AI</h1><p>Body that is reasonably long.</p></body></html>'
        )
        out = extract_from_html(html, source_url="https://x/jobs/1")
        assert out.posted_at is not None
        assert out.posted_at.isoformat() == "2026-05-21"
        assert out.posted_at_source == "meta_published_time"

    def test_empty_html_returns_empty_extracted(self):
        out = extract_from_html("", source_url="https://x/jobs/1")
        assert out.title is None
        assert out.body_text == ""


class TestExpiredSignals:
    def test_detects_known_expired_phrases(self):
        for phrase in EXPIRED_PHRASES:
            html = f"<html><body><h1>VP AI</h1><p>{phrase}.</p></body></html>"
            out = extract_from_html(html, source_url="https://x/jobs/1")
            assert phrase in out.expired_signals

    def test_no_false_positives_on_normal_jobs(self):
        html = (
            "<html><body><h1>VP AI</h1>"
            "<p>This role is responsible for leading AI transformation. "
            "You will work with multiple teams.</p></body></html>"
        )
        out = extract_from_html(html, source_url="https://x/jobs/1")
        assert out.expired_signals == []


class TestSignalDetection:
    def test_email_extracted_as_recruiter_contact(self):
        html = (
            "<html><body><h1>VP AI</h1>"
            "<p>Contact jane.recruiter@acme.com for more.</p>"
            "<p>This is a substantive body to clear length checks.</p>"
            "</body></html>"
        )
        out = extract_from_html(html, source_url="https://x/jobs/1")
        assert out.recruiter_contact == "jane.recruiter@acme.com"

    def test_multiple_openings_phrase(self):
        html = (
            "<html><body><h1>VP AI</h1>"
            "<p>We have multiple openings across our pharma division.</p>"
            "<p>Substantive body text to clear minimum length checks.</p>"
            "</body></html>"
        )
        out = extract_from_html(html, source_url="https://x/jobs/1")
        assert out.multiple_openings is True

    def test_remote_work_mode_inferred(self):
        html = (
            "<html><body><h1>VP AI</h1>"
            "<p>This role is fully remote across the US.</p>"
            "<p>Substantive body text to clear minimum length checks.</p>"
            "</body></html>"
        )
        out = extract_from_html(html, source_url="https://x/jobs/1")
        assert out.work_mode == "remote"
