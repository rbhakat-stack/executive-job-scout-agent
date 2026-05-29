# Executive Job Scout Agent - container image
# Pinned to Python 3.11; the codebase targets 3.11+ and tests run under 3.13.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System packages required by lxml + reportlab. Pinned to slim variants.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for a fresher layer cache.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app source.
COPY app.py ./
COPY job_scout/ ./job_scout/
COPY supabase/ ./supabase/
COPY evals/ ./evals/
COPY pytest.ini ./
COPY README.md ./
COPY docs/ ./docs/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
