"""FastAPI app. Run with: uvicorn crs.api.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from crs.api.dependencies import build_engine_registry
from crs.api.routes import chat, health, audio
from crs.config import get_settings
from crs.data.loaders import DatasetLoader
from crs.retrieval.vector_store import VectorStore
from crs.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot-time setup: load data, vector store, engines."""
    configure_logging()
    settings = get_settings()
    logger.info("Starting API (provider=%s, model=%s, default_engine=%s)",
                settings.llm_provider, settings.llm_model, settings.default_engine)

    # dataset — always needed
    loader = DatasetLoader(settings)
    _ = loader.item_map
    _ = loader.user_index

    # vector store (optional — needs the build script to have run)
    vector_store: VectorStore | None = None
    try:
        vs = VectorStore(settings=settings)
        vs.load()
        vector_store = vs
        logger.info("Vector store loaded.")
    except FileNotFoundError:
        logger.warning(
            "Vector store not found on disk. RAG and Agent engines disabled. "
            "Run scripts/02_build_index.py to enable them."
        )

    # engines
    engines = build_engine_registry(
        settings=settings, loader=loader, vector_store=vector_store
    )
    logger.info("Registered engines: %s", sorted(engines.keys()))

    # stash on app.state for dependency injection
    app.state.settings = settings
    app.state.loader = loader
    app.state.vector_store = vector_store
    app.state.engines = engines

    yield

    # Nothing to tear down — all resources are in-memory
    logger.info("API shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="LLM Movie CRS",
        description="Conversational movie recommender built on LLM-Redial.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(audio.router)
    
    # Mount frontend static files
    from pathlib import Path
    static_dir = Path(__file__).parent.parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/")
    async def index():
        return RedirectResponse(url="/static/index.html")

    return app


app = create_app()
