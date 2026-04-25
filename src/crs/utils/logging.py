"""Structured logging setup.

Uses stdlib logging configured for JSON-ish single-line output so logs are
grep-able in production and readable locally.
"""
from __future__ import annotations

import logging
import sys
from typing import Final

_CONFIGURED: bool = False

_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)


def configure_logging(level: str | int = "INFO") -> None:
    """Idempotent logging configuration for the whole app."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Fetch a module-scoped logger."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
