"""Shared enums and primitive types used across schemas.

Kept in one module to avoid circular imports between agent IO schemas.
"""
from enum import Enum


class SeniorityLevel(str, Enum):
    INDIVIDUAL_CONTRIBUTOR = "individual_contributor"
    MANAGER = "manager"
    SENIOR_MANAGER = "senior_manager"
    DIRECTOR = "director"
    SENIOR_DIRECTOR = "senior_director"
    VP = "vp"
    SVP = "svp"
    EVP = "evp"
    C_SUITE = "c_suite"
    PARTNER = "partner"
    MANAGING_DIRECTOR = "managing_director"


class WorkMode(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class ATS(str, Enum):
    """Applicant Tracking Systems we have first-class extractors for."""
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    WORKDAY = "workday"
    ASHBY = "ashby"
    SMARTRECRUITERS = "smartrecruiters"
    ICIMS = "icims"
    OTHER = "other"


class FreshnessLabel(str, Enum):
    """How fresh the posting is. `UNKNOWN` is the safe default when no
    evidence-backed date can be established. The system must NEVER label a
    posting `RECENT` without a citation.
    """
    RECENT = "recent"          # within criteria.max_age_days, evidence-backed
    OLDER = "older"            # outside max_age_days but date is known
    UNKNOWN = "unknown"        # no evidence-backed date


class JobStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    REMOVED = "removed"


class SearchStrategy(str, Enum):
    EXACT_TITLE = "exact_title"
    ADJACENT_TITLE = "adjacent_title"
    INDUSTRY_AND_ROLE = "industry_and_role"
    COMPANY_AND_ROLE = "company_and_role"
    SKILL_AND_ROLE = "skill_and_role"
    ATS_SCOPED = "ats_scoped"
    URGENT_HIRING = "urgent_hiring"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


class SearchProviderName(str, Enum):
    TAVILY = "tavily"
    FAKE = "fake"
