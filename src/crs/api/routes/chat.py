"""Chat endpoints.

- ``POST /chat/stream`` — SSE stream of tokens + a final recommendation event.
- ``POST /chat`` — non-streaming JSON response for simpler clients/tests.

Both accept the same ``ChatRequest`` body. The engine is selected by the
``engine`` field in the body, falling back to the configured default.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from crs.api.dependencies import get_loader
from crs.api.streaming import stream_engine_response
from crs.crs_engines.base import BaseCRS, EngineContext
from crs.data.loaders import DatasetLoader
from crs.schemas import ChatRequest, EngineResponse, UserProfile

router = APIRouter(prefix="/chat", tags=["chat"])


def _resolve_engine(request: Request, name: str | None) -> BaseCRS:
    """Pick an engine from app.state by name, else use default."""
    settings = request.app.state.settings
    registry: dict[str, BaseCRS] = request.app.state.engines
    chosen = name or settings.default_engine
    engine = registry.get(chosen)
    if engine is None:
        raise HTTPException(
            400,
            detail=(
                f"Engine '{chosen}' not available. "
                f"Registered: {sorted(registry.keys())}"
            ),
        )
    return engine


def _build_context(req: ChatRequest, loader: DatasetLoader) -> EngineContext:
    profile: UserProfile | None = None
    if req.user_id:
        profile = loader.get_user_profile(req.user_id)
    return EngineContext(
        message=req.message,
        history=req.history,
        profile=profile,
        top_k=req.top_k,
    )


@router.post("/stream", summary="Stream chat response (SSE)")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    loader: DatasetLoader = Depends(get_loader),
) -> StreamingResponse:
    engine = _resolve_engine(request, req.engine)
    ctx = _build_context(req, loader)
    return StreamingResponse(
        stream_engine_response(engine, ctx, loader),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("", response_model=EngineResponse, summary="Chat (non-streaming)")
async def chat_once(
    req: ChatRequest,
    request: Request,
    loader: DatasetLoader = Depends(get_loader),
) -> EngineResponse:
    engine = _resolve_engine(request, req.engine)
    ctx = _build_context(req, loader)
    return await engine.recommend(ctx)
