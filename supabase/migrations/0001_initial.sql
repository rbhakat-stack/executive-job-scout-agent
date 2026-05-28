-- ============================================================================
-- Executive Job Scout Agent - initial schema
-- ============================================================================
-- Matches docs/DATA_MODEL.md. Run this once against your Supabase project
-- (SQL editor or `supabase db push`).
--
-- Conventions:
--   * `id` columns are uuid with gen_random_uuid() defaults
--   * timestamptz defaults to now() at row creation
--   * Stable string fields (status, freshness, ats) use text columns rather
--     than Postgres enums so we can evolve them without migrations.
-- ============================================================================

create extension if not exists "pgcrypto";  -- gen_random_uuid

-- ---------------------------------------------------------------------------
-- profiles: extracted candidate profile (one row per uploaded CV + LinkedIn)
-- ---------------------------------------------------------------------------
create table if not exists profiles (
    id                  uuid primary key default gen_random_uuid(),
    created_at          timestamptz not null default now(),
    resume_filename     text,
    resume_text_sha256  text not null,
    resume_text         text not null,
    linkedin_url        text,
    linkedin_text       text,
    extracted           jsonb not null,
    seniority_level     text,
    industries          text[]      not null default '{}',
    target_titles       text[]      not null default '{}'
);

create index if not exists idx_profiles_resume_text_sha256
    on profiles (resume_text_sha256);

-- ---------------------------------------------------------------------------
-- jobs: globally-deduplicated unique postings
-- ---------------------------------------------------------------------------
create table if not exists jobs (
    id                  uuid primary key default gen_random_uuid(),
    dedup_hash          text not null unique,
    canonical_url       text not null,
    apply_url           text not null,
    source_url          text not null,
    title               text not null,
    company             text not null,
    location            text,
    work_mode           text,                     -- remote | hybrid | onsite | unknown
    posted_at           date,
    freshness           text not null,            -- recent | older | unknown
    freshness_evidence  jsonb,
    body_text           text not null,
    ats                 text,                     -- greenhouse | lever | workday | ashby | smartrecruiters | icims | other
    first_seen_at       timestamptz not null default now(),
    last_checked_at     timestamptz not null default now(),
    status              text not null default 'active',  -- active | closed | removed
    signals             jsonb not null default '{}'::jsonb
);

create index if not exists idx_jobs_posted_at        on jobs (posted_at desc);
create index if not exists idx_jobs_freshness        on jobs (freshness);
create index if not exists idx_jobs_company          on jobs (company);
create index if not exists idx_jobs_location         on jobs (location);
create index if not exists idx_jobs_status           on jobs (status);
create index if not exists idx_jobs_last_checked_at  on jobs (last_checked_at);

-- ---------------------------------------------------------------------------
-- runs: one row per "Run search" action
-- ---------------------------------------------------------------------------
create table if not exists runs (
    id                  uuid primary key default gen_random_uuid(),
    created_at          timestamptz not null default now(),
    profile_id          uuid references profiles(id) on delete set null,
    criteria            jsonb not null,
    plan                jsonb,
    llm_provider        text,
    llm_model           text,
    search_provider     text,
    latency_ms          integer not null default 0,
    tokens_in           integer not null default 0,
    tokens_out          integer not null default 0,
    cost_usd            numeric(10, 4) not null default 0,
    discovered          integer not null default 0,
    validated           integer not null default 0,
    surfaced            integer not null default 0,
    rejection_log       jsonb not null default '[]'::jsonb
);

create index if not exists idx_runs_profile_id on runs (profile_id);
create index if not exists idx_runs_created_at on runs (created_at desc);

-- ---------------------------------------------------------------------------
-- run_jobs: which jobs were surfaced for which run + per-run scores/evidence
-- ---------------------------------------------------------------------------
create table if not exists run_jobs (
    run_id              uuid not null references runs(id) on delete cascade,
    job_id              uuid not null references jobs(id) on delete cascade,
    match_score         integer not null check (match_score between 0 and 100),
    urgency_score       integer not null check (urgency_score between 0 and 100),
    match_rationale     text   not null,
    concerns            text,
    application_angle   text,
    outreach_angle      text,
    evidence            jsonb  not null default '[]'::jsonb,
    red_team_decision   text   not null default 'accept',  -- accept | reject
    red_team_reasons    text[] not null default '{}',
    primary key (run_id, job_id)
);

create index if not exists idx_run_jobs_run_id        on run_jobs (run_id);
create index if not exists idx_run_jobs_job_id        on run_jobs (job_id);
create index if not exists idx_run_jobs_match_score   on run_jobs (match_score desc);
create index if not exists idx_run_jobs_urgency_score on run_jobs (urgency_score desc);

-- ---------------------------------------------------------------------------
-- user_actions: favorites, applied flag, notes (per (profile, job))
-- ---------------------------------------------------------------------------
create table if not exists user_actions (
    id            uuid primary key default gen_random_uuid(),
    job_id        uuid not null references jobs(id) on delete cascade,
    profile_id    uuid not null references profiles(id) on delete cascade,
    favorited     boolean not null default false,
    applied       boolean not null default false,
    applied_at    timestamptz,
    notes         text,
    updated_at    timestamptz not null default now(),
    unique (profile_id, job_id)
);

create index if not exists idx_user_actions_profile_fav
    on user_actions (profile_id, favorited);

-- ---------------------------------------------------------------------------
-- updated_at trigger for user_actions
-- ---------------------------------------------------------------------------
create or replace function set_updated_at()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists trg_user_actions_updated_at on user_actions;
create trigger trg_user_actions_updated_at
    before update on user_actions
    for each row execute function set_updated_at();
