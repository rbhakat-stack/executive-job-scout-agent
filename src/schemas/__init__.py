"""Public schema exports.

Importing from `src.schemas` is the canonical entrypoint; sub-module paths
are internal and may be reorganized.
"""
from .common import (
    ATS,
    FreshnessLabel,
    JobStatus,
    LLMProvider,
    SearchProviderName,
    SearchStrategy,
    SeniorityLevel,
    WorkMode,
)
from .criteria import SearchCriteria
from .job import (
    Citation,
    EvidenceBundle,
    Freshness,
    FreshnessEvidence,
    JobSignals,
    ValidatedJob,
    ValidationResult,
)
from .profile import CandidateProfile
from .report import (
    JobReport,
    RedTeamDecision,
    RejectionLogEntry,
    RunMetrics,
    RunRecord,
    UserActions,
)
from .scoring import ScoreFeature, ScoreResult
from .search import RawJobLead, SearchPlan, SearchQuery

__all__ = [
    # common
    "ATS",
    "FreshnessLabel",
    "JobStatus",
    "LLMProvider",
    "SearchProviderName",
    "SearchStrategy",
    "SeniorityLevel",
    "WorkMode",
    # profile
    "CandidateProfile",
    # criteria
    "SearchCriteria",
    # search
    "SearchPlan",
    "SearchQuery",
    "RawJobLead",
    # job
    "Citation",
    "EvidenceBundle",
    "Freshness",
    "FreshnessEvidence",
    "JobSignals",
    "ValidatedJob",
    "ValidationResult",
    # scoring
    "ScoreFeature",
    "ScoreResult",
    # report
    "JobReport",
    "RedTeamDecision",
    "RejectionLogEntry",
    "RunMetrics",
    "RunRecord",
    "UserActions",
]
