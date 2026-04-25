"""DI helpers — pull heavy objects from app.state for route handlers."""
from __future__ import annotations

from typing import Literal

from fastapi import HTTPException, Request

from crs.config import Settings, get_settings
from crs.crs_engines.base import BaseCRS
from crs.crs_engines.few_shot_crs import FewShotCRS
from crs.crs_engines.rag_crs import RAGCRS
from crs.data.loaders import DatasetLoader
from crs.retrieval.vector_store import VectorStore

EngineName = Literal["few_shot", "rag", "agent"]


def get_app_settings() -> Settings:
    return get_settings()


def get_loader(request: Request) -> DatasetLoader:
    loader: DatasetLoader | None = getattr(request.app.state, "loader", None)
    if loader is None:
        raise HTTPException(500, detail="Dataset loader not initialized.")
    return loader


def get_vector_store(request: Request) -> VectorStore:
    vs: VectorStore | None = getattr(request.app.state, "vector_store", None)
    if vs is None:
        raise HTTPException(
            500,
            detail=(
                "Vector store not initialized. "
                "Build it via scripts/02_build_index.py and restart the API."
            ),
        )
    return vs


def get_engine(
    request: Request,
    name: EngineName | None = None,
) -> BaseCRS:
    """Resolve engine by name, falling back to the default."""
    settings: Settings = request.app.state.settings
    chosen = name or settings.default_engine
    registry: dict[str, BaseCRS] = request.app.state.engines

    engine = registry.get(chosen)
    if engine is None:
        raise HTTPException(400, detail=f"Unknown engine: {chosen}")
    return engine


def build_engine_registry(
    settings: Settings,
    loader: DatasetLoader,
    vector_store: VectorStore | None,
) -> dict[str, BaseCRS]:
    """Instantiate every engine and return a name -> engine mapping.

    The agent engine is only registered if the vector store is available,
    because it depends on catalog search.
    """
    registry: dict[str, BaseCRS] = {
        "few_shot": FewShotCRS(loader=loader, settings=settings),
    }
    if vector_store is not None:
        # Build BM25 keyword index for hybrid retrieval
        from crs.retrieval.bm25 import BM25Index

        bm25 = BM25Index()
        bm25.build(loader)

        registry["rag"] = RAGCRS(
            vector_store=vector_store,
            loader=loader,
            settings=settings,
            bm25_index=bm25,
        )
        # Import lazily to avoid loading the agent stack when unused
        from crs.crs_engines.agent_crs import AgentCRS

        registry["agent"] = AgentCRS(
            vector_store=vector_store, loader=loader, settings=settings
        )
    return registry
