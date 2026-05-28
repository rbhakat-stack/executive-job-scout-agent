# Data model

All persistent tables live in Supabase Postgres. Pydantic models in `src/schemas/` are the in-memory contracts; the SQL migration in `supabase/migrations/0001_initial.sql` is the source of truth for the database.

## Tables

### `profiles`
Extracted candidate profile (one row per uploaded resume + LinkedIn pair).

| Column                | Type        | Notes |
|-----------------------|-------------|-------|
| `id`                  | uuid PK     | server-generated |
| `created_at`          | timestamptz | default now() |
| `resume_filename`     | text        | original filename (not the bytes) |
| `resume_text_sha256`  | text        | dedup hash of extracted text |
| `resume_text`         | text        | extracted text only — **no raw bytes stored** |
| `linkedin_url`        | text NULL   | |
| `linkedin_text`       | text NULL   | pasted or fetched |
| `extracted`           | jsonb       | full `CandidateProfile` JSON |
| `seniority_level`     | text        | denormalized for filtering |
| `industries`          | text[]      | denormalized |
| `target_titles`       | text[]      | denormalized |

### `runs`
One row per "Run search" click.

| Column            | Type         | Notes |
|-------------------|--------------|-------|
| `id`              | uuid PK      | |
| `created_at`      | timestamptz  | |
| `profile_id`      | uuid FK      | → profiles.id |
| `criteria`        | jsonb        | full `SearchCriteria` JSON |
| `plan`            | jsonb        | search plan |
| `llm_provider`    | text         | anthropic / openai / groq |
| `llm_model`       | text         | |
| `search_provider` | text         | tavily / ... |
| `latency_ms`      | int          | total |
| `tokens_in`       | int          | sum |
| `tokens_out`      | int          | sum |
| `cost_usd`        | numeric(10,4)| sum |
| `discovered`      | int          | raw leads from Search Agent |
| `validated`       | int          | passed Validation Agent |
| `surfaced`        | int          | passed Red Team |
| `rejection_log`   | jsonb        | array of `{job_url, stage, reason}` |

### `jobs`
One row per **unique** job (deduplicated). Multiple runs reference the same row.

| Column                  | Type         | Notes |
|-------------------------|--------------|-------|
| `id`                    | uuid PK      | |
| `dedup_hash`            | text UNIQUE  | sha256(canonical_url + norm_title + norm_company) |
| `canonical_url`         | text         | post-redirect URL |
| `apply_url`             | text         | |
| `source_url`            | text         | how we found it (search result, ATS page, etc.) |
| `title`                 | text         | |
| `company`               | text         | |
| `location`              | text NULL    | |
| `work_mode`             | text NULL    | remote / hybrid / onsite |
| `posted_at`             | date NULL    | best evidence-backed date or NULL |
| `freshness`             | text         | recent / older / unknown |
| `freshness_evidence`    | jsonb        | `{source, snippet, span}` |
| `body_text`             | text         | extracted job description |
| `ats`                   | text NULL    | greenhouse / lever / workday / ashby / smartrecruiters / icims |
| `first_seen_at`         | timestamptz  | default now() |
| `last_checked_at`       | timestamptz  | |
| `status`                | text         | active / closed / removed |
| `signals`               | jsonb        | recruiter contact, multiple openings, urgency phrases, etc. |

### `run_jobs`
Join table: which jobs were surfaced in which run, plus per-run scores.

| Column              | Type        | Notes |
|---------------------|-------------|-------|
| `run_id`            | uuid FK     | → runs.id |
| `job_id`            | uuid FK     | → jobs.id |
| `match_score`       | int         | 0–100 |
| `urgency_score`     | int         | 0–100 |
| `match_rationale`   | text        | LLM-written, citation-checked |
| `concerns`          | text NULL   | |
| `application_angle` | text NULL   | |
| `outreach_angle`    | text NULL   | |
| `evidence`          | jsonb       | citation spans per claim |
| `red_team_decision` | text        | accept / reject |
| `red_team_reasons`  | text[]      | |
| PRIMARY KEY         | (run_id, job_id) | |

### `user_actions`
Captures what the user does with results (favorite, applied, notes).

| Column        | Type        | Notes |
|---------------|-------------|-------|
| `id`          | uuid PK     | |
| `job_id`      | uuid FK     | |
| `profile_id`  | uuid FK     | |
| `favorited`   | bool        | default false |
| `applied`     | bool        | default false |
| `applied_at`  | timestamptz NULL | |
| `notes`       | text NULL   | |
| `updated_at`  | timestamptz | |

## Indexes
- `jobs(dedup_hash)` UNIQUE
- `jobs(posted_at DESC, freshness)`
- `jobs(company)`, `jobs(location)`
- `run_jobs(run_id)`, `run_jobs(job_id)`
- `user_actions(profile_id, favorited)`

## What is **not** stored
- Raw resume bytes.
- API keys (LLM, Tavily, Supabase service role). Session-only via `st.session_state`.
- LLM raw prompts/responses (we keep a digest + token counts; full transcripts are an opt-in M9 feature).

## Soft-delete and re-checks
- `jobs.status='closed'` is the standard mechanism; rows are not hard-deleted so the dedup hash stays useful and run history remains intact.
- A background re-check (M9) can re-validate the top N jobs every 24h.
