"""Dedup hashing + URL canonicalization tests.

These pin the contract that two visits to the same posting (whether via
different tracking links, different cases, or different sort orders of
query params) produce the same dedup_hash.
"""
from __future__ import annotations

from src.validation.dedup import canonicalize_url, compute_dedup_hash


class TestCanonicalizeUrl:
    def test_drops_fragment(self):
        assert canonicalize_url(
            "https://boards.greenhouse.io/acme/jobs/1#apply"
        ) == "https://boards.greenhouse.io/acme/jobs/1"

    def test_drops_trailing_slash(self):
        a = canonicalize_url("https://example.com/jobs/123/")
        b = canonicalize_url("https://example.com/jobs/123")
        assert a == b

    def test_keeps_root_slash(self):
        # Don't strip the slash if the entire path IS the slash.
        assert canonicalize_url("https://example.com/").endswith("/")

    def test_lowercases_host(self):
        assert canonicalize_url(
            "https://Boards.Greenhouse.IO/Acme/jobs/1"
        ) == "https://boards.greenhouse.io/Acme/jobs/1"

    def test_drops_tracking_params(self):
        assert canonicalize_url(
            "https://example.com/jobs/1?utm_source=x&utm_medium=y&gid=42"
        ) == "https://example.com/jobs/1?gid=42"

    def test_sorts_remaining_params(self):
        a = canonicalize_url("https://example.com/jobs/1?b=2&a=1")
        b = canonicalize_url("https://example.com/jobs/1?a=1&b=2")
        assert a == b

    def test_drops_default_port_https(self):
        assert canonicalize_url("https://example.com:443/jobs/1") == \
               canonicalize_url("https://example.com/jobs/1")


class TestComputeDedupHash:
    def test_same_input_same_hash(self):
        h1 = compute_dedup_hash(
            "https://boards.greenhouse.io/acme/jobs/1", "VP AI", "Acme Bio"
        )
        h2 = compute_dedup_hash(
            "https://boards.greenhouse.io/acme/jobs/1", "VP AI", "Acme Bio"
        )
        assert h1 == h2

    def test_tracking_params_dont_change_hash(self):
        a = compute_dedup_hash(
            "https://boards.greenhouse.io/acme/jobs/1?utm_source=tw",
            "VP AI",
            "Acme Bio",
        )
        b = compute_dedup_hash(
            "https://boards.greenhouse.io/acme/jobs/1",
            "VP AI",
            "Acme Bio",
        )
        assert a == b

    def test_title_case_difference_collapses(self):
        a = compute_dedup_hash("https://x/jobs/1", "VP AI", "Acme")
        b = compute_dedup_hash("https://x/jobs/1", "vp ai", "ACME")
        assert a == b

    def test_whitespace_normalized(self):
        a = compute_dedup_hash("https://x/jobs/1", "VP   AI", "  Acme  Bio ")
        b = compute_dedup_hash("https://x/jobs/1", "VP AI", "Acme Bio")
        assert a == b

    def test_different_jobs_different_hash(self):
        a = compute_dedup_hash("https://x/jobs/1", "VP AI", "Acme")
        b = compute_dedup_hash("https://x/jobs/2", "VP AI", "Acme")
        assert a != b

    def test_hash_length(self):
        h = compute_dedup_hash("https://x/jobs/1", "VP AI", "Acme")
        assert len(h) == 64  # sha256 hex
