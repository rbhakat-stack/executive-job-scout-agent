"""ProfileAgent tests using FakeLLM."""
from __future__ import annotations

import hashlib
import json

import pytest

from job_scout.agents.profile import ProfileAgent, ProfileAgentError
from job_scout.llm import FakeLLM
from job_scout.schemas import CandidateProfile, SeniorityLevel


def _canned_profile_json() -> str:
    return json.dumps(
        {
            "summary": "20+ years executive technology leader in life sciences.",
            "industries": ["life sciences", "pharma", "consulting"],
            "functional_expertise": ["AI strategy", "transformation", "P&L"],
            "technical_expertise": ["AI", "data", "cloud"],
            "transformation_themes": ["AI transformation", "digital reinvention"],
            "ai_data_cloud_experience": ["GenAI strategy", "data platforms"],
            "leadership_scope": "Led 200+ technologists across 4 regions.",
            "client_account_experience": ["Top 20 pharma"],
            "revenue_pl_scale": "$80M P&L",
            "seniority_level": "svp",
            "target_archetypes": ["Chief Digital Officer", "SVP Technology"],
            "search_keywords": ["AI transformation", "life sciences technology"],
            "ranking_keywords": ["AI", "pharma", "P&L"],
            "title_equivalents": [
                "SVP Technology",
                "Head of AI",
                "Chief Digital Officer",
            ],
        }
    )


def _resume_bytes(text: str) -> bytes:
    return text.encode("utf-8")


SUBSTANTIVE_RESUME = (
    "John Doe\nSVP Technology, Life Sciences\n"
    + "20 years leading technology transformation in pharma. " * 5
)


class TestHappyPath:
    def test_returns_valid_candidate_profile(self):
        llm = FakeLLM(responses=[_canned_profile_json()])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
        )
        assert isinstance(profile, CandidateProfile)
        assert profile.seniority_level is SeniorityLevel.SVP
        assert "AI transformation" in profile.transformation_themes
        assert profile.resume_filename == "resume.txt"

    def test_resume_text_sha256_is_computed_by_agent(self):
        llm = FakeLLM(responses=[_canned_profile_json()])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
        )
        expected = hashlib.sha256(SUBSTANTIVE_RESUME.encode("utf-8")).hexdigest()
        assert profile.resume_text_sha256 == expected

    def test_linkedin_url_is_attached(self):
        llm = FakeLLM(responses=[_canned_profile_json()])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
            linkedin_url="https://www.linkedin.com/in/janedoe",
        )
        assert profile.linkedin_url == "https://www.linkedin.com/in/janedoe"

    def test_linkedin_text_is_passed_into_prompt(self):
        llm = FakeLLM(responses=[_canned_profile_json()])
        agent = ProfileAgent(llm)
        agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
            linkedin_text=(
                "About\n20 years in pharma\n\nExperience\nSVP at Acme Bio\n"
            ),
        )
        assert len(llm.calls) == 1
        _system, user = llm.calls[0]
        assert "About" in user or "about" in user
        assert "Acme Bio" in user


class TestProvenanceLockdown:
    """The LLM must not be able to choose resume_filename / hash / linkedin_url."""

    def test_llm_supplied_filename_is_overridden(self):
        bad = json.loads(_canned_profile_json())
        bad["resume_filename"] = "ATTACKER.txt"
        bad["resume_text_sha256"] = "0" * 64
        bad["linkedin_url"] = "https://attacker.example/in/evil"
        llm = FakeLLM(responses=[json.dumps(bad)])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="legit.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
        )
        # Filename: caller's value wins.
        assert profile.resume_filename == "legit.txt"
        # Hash: recomputed from actual resume text.
        assert (
            profile.resume_text_sha256
            == hashlib.sha256(SUBSTANTIVE_RESUME.encode("utf-8")).hexdigest()
        )
        # LinkedIn URL: caller passed none, LLM-invented URL is stripped.
        assert profile.linkedin_url is None


class TestErrors:
    def test_too_short_resume_is_rejected(self):
        llm = FakeLLM(responses=[_canned_profile_json()])
        agent = ProfileAgent(llm)
        with pytest.raises(ProfileAgentError, match="too short"):
            agent.extract(resume_filename="r.txt", resume_bytes=b"tiny")

    def test_invalid_json_from_llm_is_rejected(self):
        llm = FakeLLM(responses=["not json at all"])
        agent = ProfileAgent(llm)
        with pytest.raises(ProfileAgentError, match="valid JSON"):
            agent.extract(
                resume_filename="resume.txt",
                resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
            )

    def test_non_object_json_is_rejected(self):
        llm = FakeLLM(responses=["[\"a list\"]"])
        agent = ProfileAgent(llm)
        with pytest.raises(ProfileAgentError, match="non-object"):
            agent.extract(
                resume_filename="resume.txt",
                resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
            )

    def test_schema_validation_failure_surfaces(self):
        # Invalid seniority value AND missing required fields - even after
        # the agent's one retry, the response is still unsalvageable.
        bogus = json.dumps(
            {"summary": "x" * 30, "seniority_level": "supreme_overlord"}
        )
        llm = FakeLLM(responses=[bogus, bogus])  # bad on both attempts
        agent = ProfileAgent(llm)
        with pytest.raises(ProfileAgentError, match="schema validation"):
            agent.extract(
                resume_filename="resume.txt",
                resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
            )

    def test_unknown_field_from_llm_is_rejected(self):
        # CandidateProfile has extra='forbid' — surfacing this catches drift.
        bad = json.loads(_canned_profile_json())
        bad["hallucinated_field"] = "nope"
        # Same bad payload returned on retry too -> still rejected.
        llm = FakeLLM(responses=[json.dumps(bad), json.dumps(bad)])
        agent = ProfileAgent(llm)
        with pytest.raises(ProfileAgentError, match="schema validation"):
            agent.extract(
                resume_filename="resume.txt",
                resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
            )


class TestRobustJSONExtraction:
    """Real LLMs sometimes wrap JSON in markdown fences or surrounding
    prose despite instructions not to. The agent should tolerate the
    common cases instead of failing the user's run."""

    def test_markdown_json_fence_is_stripped(self):
        canned = _canned_profile_json()
        wrapped = f"```json\n{canned}\n```"
        llm = FakeLLM(responses=[wrapped])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
        )
        assert profile.seniority_level is SeniorityLevel.SVP

    def test_plain_markdown_fence_is_stripped(self):
        canned = _canned_profile_json()
        wrapped = f"```\n{canned}\n```"
        llm = FakeLLM(responses=[wrapped])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
        )
        assert profile.seniority_level is SeniorityLevel.SVP

    def test_prose_around_json_is_tolerated(self):
        canned = _canned_profile_json()
        wrapped = f"Here is the JSON object you requested:\n\n{canned}\n\nLet me know if you need changes."
        llm = FakeLLM(responses=[wrapped])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
        )
        assert profile.seniority_level is SeniorityLevel.SVP

    def test_braces_inside_strings_dont_confuse_depth_counter(self):
        # JSON where a string value contains literal { and } — the
        # bracket-matching extractor must not stop early.
        from job_scout.agents.profile import _extract_json_object
        import json as _json

        tricky = '{"summary": "a } and { inside", "seniority_level": "svp"}'
        out = _extract_json_object(f"Sure, here you go:\n{tricky}\nThanks.")
        parsed = _json.loads(out)
        assert parsed["seniority_level"] == "svp"


class TestSeniorityNormalization:
    """The LLM frequently emits seniority values that mirror the resume's
    title (e.g. 'senior_partner') rather than the canonical enum value
    ('partner'). The agent normalizes common aliases BEFORE validation.
    """

    @pytest.mark.parametrize(
        "llm_value,expected_enum",
        [
            ("senior_partner",            SeniorityLevel.PARTNER),
            ("principal_partner",         SeniorityLevel.PARTNER),
            ("managing_partner",          SeniorityLevel.MANAGING_DIRECTOR),
            ("vice_president",            SeniorityLevel.VP),
            ("senior_vice_president",     SeniorityLevel.SVP),
            ("executive_vice_president",  SeniorityLevel.EVP),
            ("chief_executive_officer",   SeniorityLevel.C_SUITE),
            ("ceo",                       SeniorityLevel.C_SUITE),
            ("chief_digital_officer",     SeniorityLevel.C_SUITE),
            ("Senior Partner",            SeniorityLevel.PARTNER),  # case + spaces
            ("Senior-Partner",            SeniorityLevel.PARTNER),  # hyphen
        ],
    )
    def test_alias_is_normalized(self, llm_value, expected_enum):
        obj = json.loads(_canned_profile_json())
        obj["seniority_level"] = llm_value
        llm = FakeLLM(responses=[json.dumps(obj)])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
        )
        assert profile.seniority_level is expected_enum

    def test_canonical_value_passes_through(self):
        obj = json.loads(_canned_profile_json())
        obj["seniority_level"] = "svp"
        llm = FakeLLM(responses=[json.dumps(obj)])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
        )
        assert profile.seniority_level is SeniorityLevel.SVP


class TestRetryOnValidationFailure:
    """If the LLM emits an unsalvageable value, the agent retries once with
    the Pydantic error appended so the LLM can self-correct.
    """

    def test_retries_once_and_succeeds(self):
        # First response: invalid seniority that does NOT match any alias.
        bad = json.loads(_canned_profile_json())
        bad["seniority_level"] = "supreme_overlord"
        # Second response: valid.
        good = _canned_profile_json()
        llm = FakeLLM(responses=[json.dumps(bad), good])
        agent = ProfileAgent(llm)
        profile = agent.extract(
            resume_filename="resume.txt",
            resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
        )
        assert profile.seniority_level is SeniorityLevel.SVP
        assert len(llm.calls) == 2
        # The retry prompt should include the validation error context.
        assert "schema validation" in llm.calls[1][1] or "failed" in llm.calls[1][1].lower()

    def test_second_retry_failure_raises(self):
        bad1 = json.loads(_canned_profile_json())
        bad1["seniority_level"] = "supreme_overlord"
        bad2 = json.loads(_canned_profile_json())
        bad2["seniority_level"] = "ultra_chief"
        llm = FakeLLM(responses=[json.dumps(bad1), json.dumps(bad2)])
        agent = ProfileAgent(llm)
        with pytest.raises(ProfileAgentError, match="schema validation"):
            agent.extract(
                resume_filename="resume.txt",
                resume_bytes=_resume_bytes(SUBSTANTIVE_RESUME),
            )
