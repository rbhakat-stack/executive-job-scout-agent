"""Profile Agent.

Given a resume upload (PDF/DOCX/TXT bytes + filename) and optional LinkedIn
URL and/or pasted text, produces a validated `CandidateProfile`.

Provenance fields (`resume_filename`, `resume_text_sha256`, `linkedin_url`)
are set *authoritatively* by the agent — the LLM is not allowed to choose
them. This is enforced by overwriting any LLM-provided value before schema
validation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from pydantic import ValidationError

from src.llm import LLM
from src.parsers.linkedin import parse_linkedin_text
from src.parsers.resume import extract_resume_text
from src.schemas import CandidateProfile
from src.schemas.common import SeniorityLevel


PROFILE_SYSTEM_PROMPT = (
    "You are a profile extraction agent for an executive job-search system. "
    "Given a resume and (optional) LinkedIn profile text, you produce a strict "
    "JSON object describing the candidate. "
    "You MUST output ONLY a single JSON object (no surrounding prose, no "
    "markdown fences). Use the field names listed in the user message. "
    "Do not invent fields or facts. If a field is not supported by the "
    "source text, use an empty array or null. "
    "Do not include `resume_filename`, `resume_text_sha256`, or `linkedin_url` "
    "in your output — those are set by the calling system."
)


PROFILE_USER_TEMPLATE = """\
Schema fields you must emit (with types):
  - summary: string (2-4 sentences)
  - industries: string[]
  - functional_expertise: string[]
  - technical_expertise: string[]
  - transformation_themes: string[]
  - ai_data_cloud_experience: string[]
  - leadership_scope: string | null
  - client_account_experience: string[]
  - revenue_pl_scale: string | null
  - seniority_level: one of [{seniority_values}]
  - target_archetypes: string[]
  - search_keywords: string[]    (terms the Search Agent should inject into queries)
  - ranking_keywords: string[]   (terms the Scoring Agent should weight)
  - title_equivalents: string[]  (alternative titles that count as the same role)

Mapping guidance for seniority_level (use the closest canonical value):
  - "Senior Partner" / "Principal Partner" / "Managing Partner" -> "partner"
    (or "managing_director" for "Managing Partner" / "Managing Director")
  - "Senior Vice President" -> "svp"
  - "Executive Vice President" -> "evp"
  - "Vice President" -> "vp"
  - "Chief * Officer" (CEO, CTO, CDO, CFO, etc.) -> "c_suite"
  - "Head of *" -> map by scope: typically "vp" or "senior_director"
  - "Principal" -> "senior_director"

Resume text (verbatim):
---
{resume_text}
---

LinkedIn profile (verbatim, may be empty):
---
{linkedin_text}
---
"""

# Common seniority strings the LLM may emit that aren't enum values. Map
# them to the closest canonical value before Pydantic validation. Keys are
# normalized (lowercased, spaces/hyphens -> underscores). Values must be in
# `SeniorityLevel`.
_SENIORITY_ALIASES: dict[str, str] = {
    # Partner variants
    "senior_partner":    "partner",
    "principal_partner": "partner",
    "managing_partner":  "managing_director",
    "equity_partner":    "partner",
    # VP variants
    "vice_president":           "vp",
    "senior_vice_president":    "svp",
    "executive_vice_president": "evp",
    "vp_senior":                "svp",
    # C-suite shorthand
    "ceo": "c_suite", "cto": "c_suite", "cdo": "c_suite",
    "cfo": "c_suite", "coo": "c_suite", "ciso": "c_suite",
    "chief_executive_officer":  "c_suite",
    "chief_technology_officer": "c_suite",
    "chief_digital_officer":    "c_suite",
    "chief_financial_officer":  "c_suite",
    "chief_data_officer":       "c_suite",
    "chief_operating_officer":  "c_suite",
    # Director variants
    "principal":           "senior_director",
    "principal_director":  "senior_director",
    "executive_director":  "senior_director",
    # Manager variants
    "lead":               "senior_manager",
    "team_lead":          "senior_manager",
    # IC variants
    "ic":                 "individual_contributor",
    "individual":         "individual_contributor",
}


def _normalize_seniority_value(raw: object) -> object:
    """Map a free-form seniority string to a canonical SeniorityLevel value.

    Pass through if the value is already canonical or unrecognized — Pydantic
    will then raise the descriptive enum error and the retry path can ask the
    LLM to fix it.
    """
    if not isinstance(raw, str):
        return raw
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {v.value for v in SeniorityLevel}:
        return key  # already canonical
    return _SENIORITY_ALIASES.get(key, raw)


class ProfileAgentError(Exception):
    """The Profile Agent could not produce a valid CandidateProfile."""


# A reasonable floor for "this isn't a real resume" — covers PDFs that
# extracted to almost nothing (likely scanned image) and pasted text that's
# just a name.
_MIN_RESUME_TEXT_CHARS = 50


class ProfileAgent:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def extract(
        self,
        *,
        resume_filename: str,
        resume_bytes: bytes,
        linkedin_url: Optional[str] = None,
        linkedin_text: Optional[str] = None,
    ) -> CandidateProfile:
        resume_text = extract_resume_text(resume_filename, resume_bytes)
        if len(resume_text.strip()) < _MIN_RESUME_TEXT_CHARS:
            raise ProfileAgentError(
                "Resume text is too short to extract a profile "
                f"({len(resume_text)} chars after extraction). "
                "If the file is a scanned image, run OCR first."
            )

        linkedin_sections = parse_linkedin_text(linkedin_text)
        linkedin_blob = (
            "\n".join(f"## {k}\n{v}" for k, v in linkedin_sections.items())
            if linkedin_sections
            else "(none)"
        )

        seniority_values = ", ".join(sorted(v.value for v in SeniorityLevel))
        user_prompt = PROFILE_USER_TEMPLATE.format(
            seniority_values=seniority_values,
            resume_text=resume_text.strip(),
            linkedin_text=linkedin_blob,
        )

        response = self._llm.complete(
            system=PROFILE_SYSTEM_PROMPT,
            user=user_prompt,
        )

        raw_obj = self._parse_response(response.text)
        self._inject_provenance(raw_obj, resume_filename, resume_text, linkedin_url)

        try:
            return CandidateProfile.model_validate(raw_obj)
        except ValidationError as first_error:
            # One retry: append the validation error so the LLM can correct
            # itself. Common case: the LLM picks an enum-adjacent value
            # (e.g. 'senior_partner') that didn't normalize. Cheaper than
            # expanding the enum to every job-title variant.
            retry_prompt = (
                user_prompt
                + "\n\nYour previous output failed schema validation with:\n"
                + str(first_error)
                + "\n\nFix the offending field(s) and re-emit the entire JSON object. "
                + "Output ONLY the JSON, no prose."
            )
            retry_resp = self._llm.complete(
                system=PROFILE_SYSTEM_PROMPT,
                user=retry_prompt,
            )
            raw_obj = self._parse_response(retry_resp.text)
            self._inject_provenance(raw_obj, resume_filename, resume_text, linkedin_url)
            try:
                return CandidateProfile.model_validate(raw_obj)
            except ValidationError as second_error:
                raise ProfileAgentError(
                    f"LLM response failed schema validation: {second_error}"
                ) from second_error

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> dict:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            raise ProfileAgentError(
                f"LLM did not return valid JSON: {e}. "
                f"First 200 chars: {text[:200]!r}"
            ) from e
        if not isinstance(obj, dict):
            raise ProfileAgentError(
                f"LLM returned a non-object JSON value of type {type(obj).__name__}."
            )
        # Coerce seniority_level via the alias map BEFORE validation.
        if "seniority_level" in obj:
            obj["seniority_level"] = _normalize_seniority_value(obj["seniority_level"])
        return obj

    @staticmethod
    def _inject_provenance(
        raw_obj: dict,
        resume_filename: str,
        resume_text: str,
        linkedin_url: Optional[str],
    ) -> None:
        raw_obj["resume_filename"] = resume_filename
        raw_obj["resume_text_sha256"] = hashlib.sha256(
            resume_text.encode("utf-8")
        ).hexdigest()
        if linkedin_url:
            raw_obj["linkedin_url"] = linkedin_url
        elif "linkedin_url" in raw_obj:
            raw_obj.pop("linkedin_url", None)
