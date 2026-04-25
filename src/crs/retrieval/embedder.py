"""Sentence-transformers wrapper for producing dense embeddings.

Kept deliberately thin so swapping to a hosted API (OpenAI, Voyage) is a
one-file change. Embeddings are L2-normalized so inner-product search
equals cosine similarity.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from crs.config import Settings, get_settings
from crs.utils.logging import get_logger

logger = get_logger(__name__)


class Embedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None  # lazy-loaded

    def _load(self) -> None:
        if self._model is not None:
            return
        # Import inside method so the dependency is optional at import-time
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.settings.embedding_model)
        self._model = SentenceTransformer(self.settings.embedding_model)

    def encode(
        self, texts: list[str], batch_size: int = 64, normalize: bool = True
    ) -> np.ndarray:
        """Encode a list of texts into a (N, D) float32 numpy array."""
        self._load()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 256,
        )
        return vectors.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache(maxsize=1)
def get_default_embedder() -> Embedder:
    return Embedder()
