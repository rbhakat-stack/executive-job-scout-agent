"""Executive Job Scout Agent - Streamlit entrypoint.

The UI flow lives in `job_scout/ui/pages.py`. This file is the runtime
entry only and is intentionally defensive about path setup + import
errors because Streamlit Cloud's script runner can surface very opaque
import-system errors (e.g. a `KeyError` instead of the real cause).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Belt-and-braces: put the project root at sys.path[0] BEFORE any
# job_scout.* import. On Streamlit Cloud the script runs from
# /mount/src/<repo>/app.py and the rooted-script convention sometimes
# doesn't propagate to the import machinery.
ROOT = Path(__file__).resolve().parent
_root_str = str(ROOT)
if _root_str in sys.path:
    sys.path.remove(_root_str)
sys.path.insert(0, _root_str)


def _safe_render() -> None:
    """Import + render with a visible error surface on failure.

    Streamlit's production safety redacts uncaught exception messages,
    and the import system can wrap the original cause in a `KeyError`
    on certain hosts. Showing the full traceback in the page itself
    means we can diagnose without needing the Manage app log panel.
    """
    try:
        from job_scout.ui.pages import render_app
    except Exception as e:
        import streamlit as st
        st.set_page_config(page_title="Executive Job Scout Agent - import error")
        st.error(f"App failed to import: {type(e).__name__}: {e}")
        st.code(traceback.format_exc(), language="text")
        st.markdown(
            "**Diagnostic info**\n\n"
            f"- Python: `{sys.version}`\n"
            f"- sys.path[0]: `{sys.path[0]}`\n"
            f"- Project root: `{ROOT}`\n"
            f"- Files at root: `{sorted(p.name for p in ROOT.iterdir())[:20]}`"
        )
        return

    render_app()


_safe_render()
