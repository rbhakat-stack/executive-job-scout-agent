"""Structured logging with a graceful stdlib fallback.

Production deployments install `structlog` (in `requirements.txt`) and get
JSON output on stderr. If `structlog` isn't available — local dev installs
that haven't run `pip install -r requirements.txt`, or constrained
environments — we fall back to the stdlib `logging` module so the rest of
the pipeline still works.

Either way the interface is the same: `get_logger(name)` returns an object
with `.info(event, **kwargs)`-style structured calls.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

try:
    import structlog  # type: ignore
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False


_configured: bool = False


class _StdlibStructuredLogger:
    """Tiny stand-in for a structlog BoundLogger.

    Renders kwargs as `key=value` pairs so the output stays grep-friendly
    without depending on structlog being installed.
    """

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    @staticmethod
    def _fmt(event: str, kw: dict[str, Any]) -> str:
        if not kw:
            return event
        pairs = " ".join(f"{k}={v!r}" for k, v in kw.items())
        return f"{event} {pairs}"

    def info(self, event: str, **kw: Any) -> None:
        self._log.info(self._fmt(event, kw))

    def warning(self, event: str, **kw: Any) -> None:
        self._log.warning(self._fmt(event, kw))

    def error(self, event: str, **kw: Any) -> None:
        self._log.error(self._fmt(event, kw))

    def debug(self, event: str, **kw: Any) -> None:
        self._log.debug(self._fmt(event, kw))


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Initialize logging + structlog. Idempotent."""
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stderr, level=log_level, format="%(message)s")

    if _HAS_STRUCTLOG:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ]
        if json_output:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            cache_logger_on_first_use=True,
        )

    _configured = True


def get_logger(name: Optional[str] = None):
    if not _configured:
        configure_logging()
    n = name or "executive_job_scout"
    if _HAS_STRUCTLOG:
        return structlog.get_logger(n)
    return _StdlibStructuredLogger(n)
