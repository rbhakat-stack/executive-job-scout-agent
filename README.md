# Executive Job Scout Agent

An agentic application that helps senior executives and professionals discover **real, recently posted, validated** job opportunities aligned to their CV and LinkedIn profile.

The system is built around a strict non-negotiable: **no fake, stale, or uncited results**. Every job shown to the user must be traceable to a live source URL, validated for freshness, and accompanied by evidence-backed match and urgency explanations.

---

## Status

This repository follows an incremental, milestone-based build. See `docs/ARCHITECTURE.md` for the design and the task list for current progress.

| Milestone | Scope | State |
|-----------|-------|-------|
| M0 | Project scaffolding | Complete |
| M1 | Schemas, config, DB layer | Complete |
| M2 | Profile Agent | Complete |
| M3 | Search Agent | Complete |
| M4 | Validation + Freshness Agent | Complete |
| M5 | Scoring Agent | Complete |
| M6 | Evidence + Report Agent | Complete |
| M7 | Red Team Agent + eval suite | Complete |
| M8 | Streamlit UI | Complete |
| M9 | Observability + deployment | Complete |

**Status:** v1 build complete. 256 tests passing (1 skipped pending optional `pypdf` install). 8/8 golden eval scenarios passing.

---

## Architecture in one diagram

```
Streamlit UI
   │
   ▼
Orchestrator
   │
   ├─► Planner Agent       (search plan from criteria + profile)
   ├─► Profile Agent       (resume + LinkedIn -> CandidateProfile)
   ├─► Search Agent        (multi-strategy queries via Tavily / pluggable)
   ├─► Validation Agent    (HTTP liveness + ATS parsers + freshness)
   ├─► Scoring Agent       (match + urgency, deterministic + LLM rationale)
   ├─► Evidence Agent      (citations + text spans from job pages)
   ├─► Report Agent        (final structured output)
   └─► Red Team Agent      (rejects stale, dead, weak, uncited results)

Cross-cutting: LLM provider, HTTP fetcher, ATS extractors, Supabase repo, structured logger.
```

See `docs/ARCHITECTURE.md` and `docs/DATA_MODEL.md` for detail.

---

## Local run

> The application can be run locally without any cloud setup other than the API keys you choose to provide.

### 1. Prerequisites
- Python 3.11 or higher
- A Supabase project (URL + anon key) — see `supabase/migrations/`
- At least one of: Anthropic, OpenAI, or Groq API key
- A Tavily API key for the Search Agent

### 2. Install
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure
```powershell
Copy-Item .env.example .env
# Edit .env and fill in keys (or skip env and supply them at runtime in the Settings page)
```

### 4. Apply database migrations
Run the SQL in `supabase/migrations/` against your Supabase project (SQL editor or `supabase db push`).

### 5. Run
```powershell
streamlit run app.py
```

---

## API keys at runtime

The Settings page accepts API keys for LLM providers and the search provider. **These keys are held only in the active Streamlit session and are never persisted to the database or to disk.** A clear notice is shown next to every key field. If no key is provided at runtime, the application falls back to the corresponding environment variable.

---

## Tests and evals

```powershell
pytest                          # unit tests + golden suite (256 tests)
pytest tests/golden -m golden -v # golden-set evals only
python evals/run_evals.py        # end-to-end eval harness (pass/fail summary)
```

## Observability

The orchestrator emits structured log lines (`structlog` JSON when installed, stdlib `logging` otherwise) on every stage:

- `run_start` — criteria, model, search provider
- `plan_built` — query count
- `search_complete` — leads + provider errors
- `lead_rejected` — stage, url, reason(s)
- `run_complete` — discovered / validated / surfaced / tokens / cost

LLM token usage and cost are tracked via a `MeteredLLM` wrapper around whatever provider the user picked. Cost estimates use a static price table (`src/observability/cost.py`); update it when provider prices change. Surfaced in the UI's Results tab and in `RunRecord.metrics`.

---

## Non-negotiables (enforced in code)

- **No fake jobs.** Validation Agent must mark a job `live=true` from an HTTP fetch before it can be surfaced.
- **No stale jobs presented as active.** Freshness inference labels jobs `recent | older | unknown` based on explicit evidence — never inferred from thin air.
- **No uncited claims.** Match/urgency rationales must cite specific text spans from the source page.
- **No stored keys.** API keys live in `st.session_state` only.
- **Red Team gate.** Final outputs are filtered by `RedTeamAgent` per the rules in `docs/ARCHITECTURE.md`.

---

## Deployment

See `docs/DEPLOYMENT.md`.
