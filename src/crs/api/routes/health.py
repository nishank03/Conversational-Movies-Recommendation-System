"""Liveness and readiness probes.

- ``/healthz``: process is up (always 200 while the event loop runs).
- ``/readyz``: dependent resources are initialised (loader + at least one
  engine). Returns 503 until the app startup hook finishes.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, object]:
    state = request.app.state
    loader_ready = getattr(state, "loader", None) is not None
    engines = getattr(state, "engines", {}) or {}
    vs_ready = getattr(state, "vector_store", None) is not None

    ready = loader_ready and len(engines) > 0
    return {
        "ready": ready,
        "loader": loader_ready,
        "vector_store": vs_ready,
        "engines": sorted(engines.keys()),
    }
