# Architecture

## Goals
The system finds **real, recently posted, validated** executive roles tied to an uploaded CV and LinkedIn profile. Correctness rules dominate the design:

1. **No fake jobs.** Every surfaced job must be validated as live by an HTTP fetch.
2. **No stale jobs presented as active.** Freshness has explicit evidence or is labelled `unknown`.
3. **No uncited claims.** Match/urgency rationales must cite source text spans.
4. **No stored keys.** API keys are session-only.
5. **Red Team gate.** Final outputs pass through rejection rules before reaching the user.

## High-level diagram

```
                        ┌─────────────────────────┐
                        │      Streamlit UI       │
                        │  (src/ui/*, app.py)     │
                        └────────────┬────────────┘
                                     │
                            ┌────────▼────────┐
                            │  Orchestrator   │
                            │  (pure async    │
                            │   pipeline)     │
                            └────────┬────────┘
                                     │
   ┌───────────────┬───────────────┬─┴───────────────┬───────────────┬──────────────┐
   ▼               ▼               ▼                 ▼               ▼              ▼
┌────────┐    ┌─────────┐    ┌──────────┐      ┌────────────┐  ┌──────────┐  ┌──────────┐
│Planner │    │Profile  │    │ Search   │      │ Validation │  │ Scoring  │  │Evidence  │
│Agent   │    │Agent    │    │ Agent    │      │ Agent      │  │ Agent    │  │Agent     │
└────────┘    └─────────┘    └──────────┘      └────────────┘  └──────────┘  └──────────┘
                                                                                  │
                                                                                  ▼
                                                                          ┌──────────────┐
                                                                          │ Report Agent │
                                                                          └──────┬───────┘
                                                                                 │
                                                                          ┌──────▼───────┐
                                                                          │ Red Team     │
                                                                          │ Agent (gate) │
                                                                          └──────┬───────┘
                                                                                 │
                                                                                 ▼
                                                                          surfaced to UI
```

Cross-cutting modules:
- `src/llm/`         — pluggable LLM provider (Anthropic / OpenAI / Groq), session-only key resolution, fake provider for tests.
- `src/search_providers/` — Tavily by default; pluggable interface; fake provider for tests.
- `src/parsers/`     — Resume (PDF/DOCX/TXT), LinkedIn text, ATS HTML/JSON (Greenhouse, Lever, Workday, Ashby, SmartRecruiters, iCIMS).
- `src/db/`          — `JobRepo`, `RunRepo`, `ProfileRepo` interfaces; Supabase implementation + in-memory test impl.
- `src/observability/` — structured logging, run metrics, cost tracking hooks.

## Agent contracts (typed Pydantic IO)

| Agent       | Input                                                    | Output                                         |
|-------------|----------------------------------------------------------|-----------------------------------------------|
| Planner     | `CandidateProfile`, `SearchCriteria`                     | `SearchPlan` (list of `SearchQuery`)          |
| Profile     | resume bytes + LinkedIn text                             | `CandidateProfile`                            |
| Search      | `SearchPlan`                                             | `list[RawJobLead]`                            |
| Validation  | `RawJobLead`                                             | `ValidatedJob` with `liveness` + `freshness`  |
| Scoring     | `ValidatedJob`, `CandidateProfile`, `SearchCriteria`     | `ScoreResult` (match + urgency + rationales)  |
| Evidence    | `ValidatedJob`, `ScoreResult`                            | `EvidenceBundle` (citation spans)             |
| Report      | `ValidatedJob` + `ScoreResult` + `EvidenceBundle`        | `JobReport` (user-facing)                     |
| Red Team    | `JobReport`                                              | `RedTeamDecision` (`accept` | `reject` + reasons) |

The orchestrator threads these contracts; **no agent silently mutates shared state**. This is what makes per-step tests possible.

## Search strategies (Search Agent)

Deterministic templates derived from `CandidateProfile` and `SearchCriteria`:

1. Exact role title: `"VP AI" "life sciences"`
2. Adjacent role titles (from profile equivalents): `"Head of AI" "pharma"`
3. Industry + role: `"life sciences technology" "managing director"`
4. Company + role: `"Pfizer" "Chief Digital Officer"`
5. Skill + executive title: `"AI transformation" "SVP technology"`
6. ATS-scoped: `site:greenhouse.io "commercial technology" "pharma"`, also `site:lever.co`, `site:jobs.ashbyhq.com`, `site:smartrecruiters.com`, `site:icims.com`, Workday URL patterns.
7. Urgency-tagged: `"urgently hiring" <role>`, `"immediate start" <role>`.

Query templates live in `src/agents/search.py`; the LLM is **not** used to generate queries (deterministic, debuggable). The LLM is used downstream for rationale generation only.

## Validation pipeline (Validation Agent)

For every `RawJobLead`:

1. `HEAD` then `GET` with timeout + redirect cap. Bot-friendly UA. 4xx/5xx → reject.
2. ATS detection by URL pattern (`boards.greenhouse.io`, `jobs.lever.co`, `*.myworkdayjobs.com`, `jobs.ashbyhq.com`, `jobs.smartrecruiters.com`, `*.icims.com`).
3. ATS-specific extractor: structured JSON (Greenhouse, Lever, Ashby) or schema.org `JobPosting` JSON-LD (Workday, SmartRecruiters, iCIMS) or HTML fallback.
4. Required fields: title, company, location, body. Missing → reject.
5. Freshness:
   - Prefer ATS `updated_at` / `created_at`.
   - Then schema.org `datePosted`.
   - Then page metadata (`<meta property="article:published_time">`, etc.).
   - Then search-engine date (Tavily result).
   - If none: `freshness=unknown` (never `recent`).
6. Expiry signals: text patterns (`"this role is no longer accepting"`, `"position filled"`), redirect to careers index, missing job ID on ATS API. Any → reject.
7. Dedup hash: `sha256(canonical_url + normalized_title + normalized_company)`; collisions are merged (multi-source citation kept).

## Scoring (Scoring Agent)

Two scores, both 0–100, both have a deterministic baseline plus an LLM-written explanation that **must cite source spans**.

### Match score features
- Role seniority match (lookup table over normalized titles)
- Industry match (overlap between profile industries and posting)
- Functional expertise match (token-set overlap on weighted keywords)
- Tech/domain match (AI/data/cloud/CRM/platform)
- Leadership scope match (P&L, team size signals)
- Location/remote/travel fit (criteria vs posting)
- Compensation fit if signal exists
- Strategic adjacency (LLM-rated 0–10, clipped)

### Urgency score features
- Posted ≤ 7 days
- Posted ≤ 14 days
- "Urgently hiring" / "immediate start" language
- Recruiter contact listed
- Multiple openings
- Transformation language ("AI transformation", "digital reinvention")
- Refreshed posting
- Source reliability (ATS > aggregator > random search hit)

Each feature returns a weight + an evidence reference. The Scoring Agent rejects any LLM-written rationale that does not include the expected citation IDs.

## Red Team rules

`RedTeamAgent` rejects a `JobReport` if any of:

- Apply URL missing
- Source URL not 2xx on last check
- Posting older than `criteria.max_age_days` (default 14) unless `criteria.allow_older=true`
- Posting flagged closed/expired
- Evidence count < 1 for any non-trivial claim
- Match rationale generic (heuristic: low cosine vs role/profile keywords)
- Posting date claimed `recent` without an evidence ref
- Job not relevant to candidate profile (match < `criteria.min_match`)
- Duplicate of an already-accepted job (post-dedup safety net)

## Persistence (Supabase)

Schema details in `docs/DATA_MODEL.md`. Stored: user profile extracts, search queries, jobs, validation results, scores, evidence, dedup hashes, run telemetry. **Never stored**: API keys, raw resume bytes (we store extracted text + filename hash).

## Observability

Structured logs (`structlog`) per stage with: `run_id`, `agent`, `latency_ms`, `tokens_in`, `tokens_out`, `cost_usd`, `tool`, `result_count`, `rejected_count`, `reject_reasons`. Surfaced in a Streamlit "Run details" expander.

## Failure modes and what we do

| Failure | Behavior |
|---------|----------|
| No LLM key available | Profile/Scoring agents return deterministic-only output with a warning banner. |
| Tavily down or empty | Search Agent returns empty result set; UI shows "no leads from search provider" not fake jobs. |
| Validation finds no live jobs | UI shows "0 validated jobs" with the rejection table — never a synthetic placeholder. |
| Supabase down | App falls back to in-memory repo for the session and shows a persistence warning. |
