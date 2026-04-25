"""FAISS-backed vector store for movie retrieval.

Build-time: embed every movie, persist index + metadata to disk.
Query-time: load once, answer top-K queries in O(log N) with IndexFlatIP.
For ~10k movies IndexFlatIP is fast enough; swap to IVF if the catalog
grows past ~1M.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from crs.config import Settings, get_settings
from crs.data.enrich import build_enriched_catalog
from crs.data.loaders import DatasetLoader
from crs.retrieval.embedder import Embedder
from crs.schemas import Movie, RetrievedCandidate
from crs.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """FAISS IndexFlatIP over movie embeddings."""

    INDEX_FILENAME: str = "movies_index.faiss"
    META_FILENAME: str = "movies_meta.pkl"

    def __init__(
        self,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings)
        self._index = None
        self._movies: list[Movie] = []

    # ---------- Build ----------

    @staticmethod
    def _render_doc(title: str, description: str | None) -> str:
        if description:
            return f"{title}. {description}"
        return title

    def build(self, loader: DatasetLoader | None = None) -> None:
        """Build the FAISS index from the enriched catalog."""
        import faiss  # local import — optional dep

        loader = loader or DatasetLoader(self.settings)
        catalog = build_enriched_catalog(loader)

        docs = [
            self._render_doc(row["title"], row.get("description"))
            for _, row in catalog.iterrows()
        ]
        logger.info("Encoding %d movie documents...", len(docs))
        vectors = self.embedder.encode(docs)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._index = index

        self._movies = [
            Movie(
                item_id=row["item_id"],
                title=row["title"],
                description=row.get("description") if pd.notna(row.get("description")) else None,
            )
            for _, row in catalog.iterrows()
        ]
        logger.info("Built FAISS index with %d vectors (dim=%d)", len(self._movies), dim)

    def save(self) -> tuple[Path, Path]:
        import faiss

        if self._index is None:
            raise RuntimeError("Index not built; call build() first.")
        out_dir = self.settings.project_root / self.settings.vector_store_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        index_path = out_dir / self.INDEX_FILENAME
        meta_path = out_dir / self.META_FILENAME

        faiss.write_index(self._index, str(index_path))
        with meta_path.open("wb") as f:
            pickle.dump([m.model_dump() for m in self._movies], f)

        logger.info("Saved vector store to %s", out_dir)
        return index_path, meta_path

    # ---------- Load ----------

    def load(self) -> None:
        import faiss

        out_dir = self.settings.project_root / self.settings.vector_store_dir
        index_path = out_dir / self.INDEX_FILENAME
        meta_path = out_dir / self.META_FILENAME

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Vector store missing at {out_dir}. "
                "Run scripts/02_build_index.py first."
            )

        self._index = faiss.read_index(str(index_path))
        with meta_path.open("rb") as f:
            raw = pickle.load(f)
        self._movies = [Movie(**item) for item in raw]
        logger.info("Loaded vector store: %d movies", len(self._movies))

    # ---------- Query ----------

    def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedCandidate]:
        if self._index is None:
            raise RuntimeError("Vector store not loaded; call load() or build().")

        top_k = top_k or self.settings.retrieval_top_k
        qvec = self.embedder.encode([query])
        scores, indices = self._index.search(qvec, top_k)

        results: list[RetrievedCandidate] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self._movies):
                continue
            results.append(
                RetrievedCandidate(movie=self._movies[idx], score=float(score))
            )
        return results

    def search_batch(
        self, queries: list[str], top_k: int | None = None
    ) -> list[list[RetrievedCandidate]]:
        if self._index is None:
            raise RuntimeError("Vector store not loaded.")
        top_k = top_k or self.settings.retrieval_top_k
        qvecs = self.embedder.encode(queries)
        scores, indices = self._index.search(qvecs, top_k)

        out: list[list[RetrievedCandidate]] = []
        for row_scores, row_indices in zip(scores, indices):
            row: list[RetrievedCandidate] = []
            for score, idx in zip(row_scores.tolist(), row_indices.tolist()):
                if idx < 0 or idx >= len(self._movies):
                    continue
                row.append(
                    RetrievedCandidate(movie=self._movies[idx], score=float(score))
                )
            out.append(row)
        return out
