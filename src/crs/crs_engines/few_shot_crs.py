"""Few-shot CRS engine.

Bakes exemplar dialogues into the prompt, no retrieval. Serves as a baseline --
fast and surprisingly decent, but can hallucinate titles not in the catalog.
"""
from __future__ import annotations

from typing import AsyncIterator

from crs.config import Settings, get_settings
from crs.crs_engines.base import BaseCRS, EngineContext
from crs.data.loaders import DatasetLoader, get_default_loader
from crs.llm.client import LLMClient, get_default_client
from crs.llm.formatters import history_to_messages, render_user_profile
from crs.llm.prompts import load_system_prompt
from crs.llm.prompts.few_shot_examples import build_few_shot_block
from crs.schemas import EngineResponse
from crs.utils.logging import get_logger
from crs.utils.timing import timer

logger = get_logger(__name__)


class FewShotCRS(BaseCRS):
    """LLM + few-shot examples, no retrieval."""

    name = "few_shot"

    def __init__(
        self,
        llm: LLMClient | None = None,
        loader: DatasetLoader | None = None,
        settings: Settings | None = None,
        n_examples: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_default_client()
        self.loader = loader or get_default_loader()
        self.n_examples = n_examples

    # ---------- Prompt assembly ----------

    def _build_system_prompt(self, ctx: EngineContext) -> str:
        base = load_system_prompt(self.settings.prompt_version)
        profile_block = render_user_profile(ctx.profile)
        few_shot = build_few_shot_block(n_examples=self.n_examples)

        # Few-shot engine has NO candidate list, so we relax rule #1 at runtime.
        # We tell the model it may use its own knowledge but must still return
        # an id if it recognises the movie.
        note = (
            "NOTE: No candidate list is provided for this engine. You may "
            "draw on your general movie knowledge. When recommending, try to "
            "use a movie that matches the user's taste profile. Still emit a "
            "<REC> block; use the item id 'UNKNOWN' if the specific id cannot "
            "be determined."
        )

        parts = [base, "", profile_block, "", note]
        if few_shot:
            parts.extend(["", few_shot])
        return "\n".join(parts)

    # ---------- Non-streaming ----------

    async def recommend(self, ctx: EngineContext) -> EngineResponse:
        system = self._build_system_prompt(ctx)
        messages = history_to_messages(ctx.history, ctx.message)

        with timer() as t:
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
        system = self._build_system_prompt(ctx)
        messages = history_to_messages(ctx.history, ctx.message)

        async for chunk in self.llm.stream(messages=messages, system=system):
            yield chunk
