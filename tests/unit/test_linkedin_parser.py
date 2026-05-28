"""LinkedIn text parser tests."""
from __future__ import annotations

from src.parsers.linkedin import parse_linkedin_text


class TestEmpty:
    def test_empty_string_returns_empty_dict(self):
        assert parse_linkedin_text("") == {}

    def test_none_returns_empty_dict(self):
        assert parse_linkedin_text(None) == {}

    def test_whitespace_only_returns_empty_dict(self):
        assert parse_linkedin_text("   \n   \t  \n") == {}


class TestParsing:
    def test_no_headers_goes_to_preamble(self):
        out = parse_linkedin_text("Senior tech leader with 20 years in pharma.")
        assert list(out.keys()) == ["__preamble__"]
        assert "Senior tech leader" in out["__preamble__"]

    def test_standard_sections(self):
        raw = (
            "Jane Doe\n"
            "VP AI Transformation\n\n"
            "About\n"
            "20 years in life sciences technology.\n\n"
            "Experience\n"
            "SVP at Acme Bio - led AI transformation.\n\n"
            "Skills\n"
            "Python, AI strategy, P&L management\n"
        )
        out = parse_linkedin_text(raw)
        assert out["__preamble__"].startswith("Jane Doe")
        assert "20 years" in out["about"]
        assert "Acme Bio" in out["experience"]
        assert "Python" in out["skills"]

    def test_ampersand_section_is_canonicalized(self):
        raw = "About\nfoo\n\nHonors & awards\nbar\n"
        out = parse_linkedin_text(raw)
        assert "foo" in out["about"]
        assert "bar" in out["honors_and_awards"]

    def test_case_insensitive_headers(self):
        raw = "ABOUT\nblah\n\nEXPERIENCE\nblah blah\n"
        out = parse_linkedin_text(raw)
        assert "blah" in out["about"]
        assert "blah blah" in out["experience"]

    def test_repeated_section_concatenates(self):
        raw = "Experience\nfirst gig\n\nExperience\nsecond gig\n"
        out = parse_linkedin_text(raw)
        assert "first gig" in out["experience"]
        assert "second gig" in out["experience"]

    def test_empty_section_body_is_omitted(self):
        raw = "About\n\nExperience\nreal content\n"
        out = parse_linkedin_text(raw)
        assert "about" not in out
        assert "real content" in out["experience"]
