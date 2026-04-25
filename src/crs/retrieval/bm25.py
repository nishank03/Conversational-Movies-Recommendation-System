"""BM25 keyword index.

Dense retrieval is great for semantic matches ("something cozy and festive")
but loses on exact title lookups ("show me Inception"). BM25 is a cheap
complement. A hybrid engine can fuse the two with reciprocal rank fusion.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from crs.data.loaders import DatasetLoader
from crs.schemas import Movie, RetrievedCandidate
from crs.utils.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Simple BM25 index over movie titles (and optional descriptions)."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._movies: list[Movie] = []

    def build(self, loader: DatasetLoader) -> None:
        movies = list(loader.iter_movies())
        corpus = [_tokenize(m.title) for m in movies]
        self._bm25 = BM25Okapi(corpus)
        self._movies = movies
        logger.info("Built BM25 index over %d titles", len(movies))

    def search(self, query: str, top_k: int = 10) -> list[RetrievedCandidate]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built.")
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # argpartition-like top-k without sorting the whole array
        import numpy as np

        idx = np.argsort(-scores)[:top_k]
        out: list[RetrievedCandidate] = []
        for i in idx:
            s = float(scores[i])
            if s <= 0:
                continue
            out.append(RetrievedCandidate(movie=self._movies[i], score=s))
        return out
