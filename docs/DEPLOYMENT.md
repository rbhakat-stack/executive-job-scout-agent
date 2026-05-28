# Deployment

The application is a Streamlit app. There are three reasonable deployment surfaces; pick whichever fits your operational model.

## Option A: Streamlit Community Cloud (simplest)

1. Push this repository to GitHub.
2. At https://streamlit.io/cloud, create a new app pointing at `app.py`.
3. In the app's **Secrets** UI, paste:
   ```toml
   ANTHROPIC_API_KEY = "..."
   TAVILY_API_KEY = "..."
   SUPABASE_URL = "..."
   SUPABASE_ANON_KEY = "..."
   LLM_PROVIDER = "anthropic"
   LLM_MODEL = "claude-opus-4-7"
   SEARCH_PROVIDER = "tavily"
   APP_ENV = "prod"
   ```
4. Deploy. Subsequent pushes to `main` redeploy automatically.

Notes:
- Streamlit secrets are not committed to git and are available at runtime under `st.secrets`.
- Users can still override LLM and search keys from the Settings page (session-only), useful for shared deployments where each user supplies their own key.

## Option B: Container on a small VM / Cloud Run / Render / Fly

The repository ships a `Dockerfile` and `.dockerignore`. Build and run:

```powershell
docker build -t executive-job-scout .

docker run --rm -p 8501:8501 `
  -e ANTHROPIC_API_KEY=... `
  -e TAVILY_API_KEY=... `
  -e SUPABASE_URL=... `
  -e SUPABASE_ANON_KEY=... `
  -e LLM_PROVIDER=anthropic `
  -e LLM_MODEL=claude-opus-4-7 `
  executive-job-scout
```

The image is pinned to Python 3.11, runs Streamlit headless, exposes 8501, and includes a `/_stcore/health` healthcheck. The `Dockerfile` installs the few system libs (`libxml2-dev`, `libxslt1-dev`) that `lxml` and `reportlab` depend on.

Pass secrets via environment variables (see `.env.example`). Users can still override LLM and Tavily keys at runtime in the Settings page — session-only, never persisted.

## Option C: Local desktop run

```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py
```

The local run is the canonical development surface; everything is designed to work without cloud infrastructure other than the API keys you choose to use.

## Supabase setup

1. Create a project at https://supabase.com.
2. From **Project Settings -> API**, copy:
   - Project URL -> `SUPABASE_URL`
   - `anon` public key -> `SUPABASE_ANON_KEY`
3. In the SQL editor, paste and run `supabase/migrations/0001_initial.sql` (lands in M1).
4. (Optional) Configure Row Level Security policies if multiple users will share the same Supabase project.

## Secrets management — golden rules

- **Never** commit `.env`, `.streamlit/secrets.toml`, or any file containing real keys.
- **Never** store user-supplied LLM/search keys in the database. The UI Settings page keeps them in `st.session_state` for the active session only.
- **Service-role keys** (Supabase) are server-only. Do not expose them in client-rendered surfaces.

## Health and rollback

- The Streamlit health endpoint `/_stcore/health` returns 200 when the app is up.
- For Streamlit Cloud, rollback = redeploy a previous commit from the dashboard.
- For container hosts, keep two tags (`:current`, `:previous`) and flip traffic.
