"""Prompt loader.

Prompts live as .txt files next to this module so they diff cleanly in git
and can be swapped by version string (v1/v2/v3) without touching code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache(maxsize=8)
def load_system_prompt(version: str) -> str:
    path = _DIR / f"system_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No such prompt file: {path}")
    return path.read_text(encoding="utf-8")
