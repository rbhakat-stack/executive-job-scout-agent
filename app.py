"""Executive Job Scout Agent - Streamlit entrypoint.

The UI flow lives in `src/ui/pages.py`. This file is the runtime entry only.
"""
from __future__ import annotations

from job_scout.ui.pages import render_app


if __name__ == "__main__":
    render_app()
else:
    # Streamlit imports the script rather than running it as __main__.
    render_app()
