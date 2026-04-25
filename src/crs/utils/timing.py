"""Lightweight timing helpers for latency tracking."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager that records elapsed time in milliseconds.

    Usage:
        with timer() as t:
            do_work()
        print(t["elapsed_ms"])
    """
    result: dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0


def now_ms() -> float:
    """Monotonic timestamp in milliseconds."""
    return time.perf_counter() * 1000.0
