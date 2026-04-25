"""Agent-based Conversational Recommender.

Strategy: a ReAct-style loop where the LLM can call tools to search the
catalog, look up a user's history, or fetch movie info. The LLM plans
(iterates) until it is confident enough to recommend.

This is the most powerful but also the slowest and most expensive engine.
Best for messy user queries that need multiple retrieval passes.
"""
from __future__ import annotations

from typing import AsyncIterator

from crs.agents.orchestrator import AgentOrchestrator
from crs.config import Settings, get_settings
from crs.crs_engines.base import BaseCRS, EngineContext
from crs.data.loaders import DatasetLoader, get_default_loader
from crs.llm.client import LLMClient, get_default_client
from crs.llm.prompts import load_system_prompt
from crs.retrieval.vector_store import VectorStore
from crs.schemas import EngineResponse
from crs.utils.logging import get_logger
from crs.utils.timing import timer

logger = get_logger(__name__)


class AgentCRS(BaseCRS):
    """Tool-using agent for movie recommendation."""

    name = "agent"

    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMClient | None = None,
        loader: DatasetLoader | None = None,
        settings: Settings | None = None,
        max_iterations: int = 4,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_default_client()
        self.loader = loader or get_default_loader()
        self.vector_store = vector_store
        self.orchestrator = AgentOrchestrator(
            llm=self.llm,
            loader=self.loader,
            vector_store=self.vector_store,
            max_iterations=max_iterations,
        )

    def _system_prompt(self) -> str:
        base = load_system_prompt(self.settings.prompt_version)
        agent_note = (
            "You have access to tools for searching the catalog and looking "
            "up user history. Use them before committing to a recommendation "
            "when you are unsure. Always recommend only movies that appear in "
            "tool results."
        )
        return f"{base}\n\n{agent_note}"

    async def recommend(self, ctx: EngineContext) -> EngineResponse:
        with timer() as t:
            result = await self.orchestrator.run(
                ctx=ctx, system_prompt=self._system_prompt()
            )
        raw = self.strip_thinking(result)
        recs = self.parse_recommendations(raw, self.loader.item_map)
        reply = self.strip_rec_block(raw)
        return EngineResponse(
            reply=reply,
            recommendations=recs,
            engine=self.name,
            prompt_version=self.settings.prompt_version,
            latency_ms=t["elapsed_ms"],
        )

    async def stream(self, ctx: EngineContext) -> AsyncIterator[str]:
        """Agent streaming: execute the tool loop, then stream the final reply.

        True token-level streaming through a tool loop is complex; here we
        stream the final answer only. For production, consider emitting
        status events ("searching...", "comparing...") during tool calls.
        """
        result = await self.orchestrator.run(
            ctx=ctx, system_prompt=self._system_prompt()
        )
        # Chunk the result to preserve the streaming contract
        chunk_size = 40
        for i in range(0, len(result), chunk_size):
            yield result[i : i + chunk_size]
