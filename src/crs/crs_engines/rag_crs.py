"""RAG-based CRS engine.

Retrieves top-K candidates via FAISS, injects them into the prompt,
and asks the LLM to pick from that list only. Works well on closed catalogs.

Optionally fuses dense (FAISS) and sparse (BM25) retrieval via reciprocal
rank fusion for better coverage of both semantic and exact-title queries.
"""
from __future__ import annotations

from typing import AsyncIterator

from crs.config import Settings, get_settings
from crs.crs_engines.base import BaseCRS, EngineContext
from crs.data.loaders import DatasetLoader, get_default_loader
from crs.llm.client import LLMClient, get_default_client
from crs.llm.formatters import (
    history_to_messages,
    render_candidates,
    render_user_profile,
)
from crs.llm.prompts import load_system_prompt
from crs.retrieval.bm25 import BM25Index
from crs.retrieval.vector_store import VectorStore
from crs.schemas import EngineResponse, RetrievedCandidate, UserProfile
from crs.utils.logging import get_logger
from crs.utils.timing import timer

logger = get_logger(__name__)


class RAGCRS(BaseCRS):
    """Retrieval-Augmented CRS using a FAISS vector store."""

    name = "rag"

    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMClient | None = None,
        loader: DatasetLoader | None = None,
        settings: Settings | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_default_client()
        self.loader = loader or get_default_loader()
        self.vector_store = vector_store
        self.bm25 = bm25_index

    # ---------- Retrieval query construction ----------

    @staticmethod
    def _profile_hint(profile: UserProfile | None, max_titles: int = 8) -> str:
        if profile is None or not profile.history:
            return ""
        recent = profile.history[:max_titles]
        return " Tastes (past watches): " + ", ".join(m.title for m in recent)

    def _build_query(self, ctx: EngineContext) -> str:
        """Combine current message, recent assistant turn, and history cues.

        Using the last assistant turn helps when the user says things like
        "tell me more" — the message alone is too sparse to retrieve well.
        """
        parts: list[str] = [ctx.message]

        if ctx.history:
            # Include the last assistant turn for disambiguation
            for m in reversed(ctx.history):
                if m.role == "assistant":
                    parts.append(m.content[:400])
                    break
            # And the most recent user turn before the current one
            prev_user = [m for m in ctx.history if m.role == "user"]
            if prev_user:
                parts.append(prev_user[-1].content[:400])

        parts.append(self._profile_hint(ctx.profile))
        return " ".join(p for p in parts if p).strip()

    # ---------- Candidate filtering ----------

    def _filter_candidates(
        self,
        candidates: list[RetrievedCandidate],
        profile: UserProfile | None,
    ) -> list[RetrievedCandidate]:
        """Drop candidates already in the user's watch history."""
        if profile is None or not profile.history:
            return candidates
        seen_ids = {m.item_id for m in profile.history}
        return [c for c in candidates if c.movie.item_id not in seen_ids]

    # ---------- Prompt assembly ----------

    def _build_system_prompt(
        self,
        ctx: EngineContext,
        candidates: list[RetrievedCandidate],
    ) -> str:
        base = load_system_prompt(self.settings.prompt_version)
        profile_block = render_user_profile(ctx.profile)
        candidate_block = render_candidates(
            candidates, max_items=self.settings.rerank_top_k
        )
        return "\n\n".join([base, profile_block, candidate_block])

    # ---------- Retrieval step (shared by stream + recommend) ----------

    def _retrieve(self, ctx: EngineContext) -> list[RetrievedCandidate]:
        top_k = ctx.top_k or self.settings.retrieval_top_k
        query = self._build_query(ctx)
        logger.debug("RAG retrieval query: %s", query[:200])

        dense = self.vector_store.search(query, top_k=top_k)

        # Hybrid: fuse dense (FAISS) + sparse (BM25) via reciprocal rank
        if self.bm25 is not None:
            sparse = self.bm25.search(query, top_k=top_k)
            merged = self._reciprocal_rank_fusion(dense, sparse)
        else:
            merged = dense

        filtered = self._filter_candidates(merged, ctx.profile)
        return filtered[: self.settings.rerank_top_k]

    @staticmethod
    def _reciprocal_rank_fusion(
        dense: list[RetrievedCandidate],
        sparse: list[RetrievedCandidate],
        k: int = 60,
    ) -> list[RetrievedCandidate]:
        """Merge two ranked lists using reciprocal rank fusion (RRF).

        RRF score = Σ 1/(k + rank_i) across each list.  ``k`` is a
        smoothing constant (default 60, per the original paper).
        """
        scores: dict[str, float] = {}
        movie_map: dict[str, RetrievedCandidate] = {}

        for rank, c in enumerate(dense, start=1):
            mid = c.movie.item_id
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
            movie_map[mid] = c

        for rank, c in enumerate(sparse, start=1):
            mid = c.movie.item_id
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
            if mid not in movie_map:
                movie_map[mid] = c

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [
            RetrievedCandidate(movie=movie_map[mid].movie, score=scores[mid])
            for mid in sorted_ids
        ]

    # ---------- Non-streaming ----------

    async def recommend(self, ctx: EngineContext) -> EngineResponse:
        with timer() as t:
            candidates = self._retrieve(ctx)
            system = self._build_system_prompt(ctx, candidates)
            messages = history_to_messages(ctx.history, ctx.message)
            raw = await self.llm.complete(messages=messages, system=system)

        raw = self.strip_thinking(raw)
        recs = self.parse_recommendations(raw, self.loader.item_map)
        reply = self.strip_rec_block(raw)

        return EngineResponse(
            reply=reply,
            recommendations=recs,
            engine=self.name,
            prompt_version=self.settings.prompt_version,
            latency_ms=t["elapsed_ms"],
        )

    # ---------- Streaming ----------

    async def stream(self, ctx: EngineContext) -> AsyncIterator[str]:
        candidates = self._retrieve(ctx)
        system = self._build_system_prompt(ctx, candidates)
        messages = history_to_messages(ctx.history, ctx.message)

        async for chunk in self.llm.stream(messages=messages, system=system):
            yield chunk
