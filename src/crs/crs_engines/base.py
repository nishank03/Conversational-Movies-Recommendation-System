"""Base class for recommendation engines + shared parsing helpers."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from crs.schemas import EngineResponse, Message, Recommendation, UserProfile


# Matches <REC>id1, id2</REC> blocks emitted by the model
_REC_BLOCK = re.compile(r"<REC>\s*([^<]+?)\s*</REC>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class EngineContext:
    """All per-request data an engine needs."""

    message: str
    history: list[Message]
    profile: UserProfile | None = None
    top_k: int | None = None


class BaseCRS(ABC):
    """Abstract conversational recommender engine."""

    name: str = "base"

    @abstractmethod
    async def recommend(self, ctx: EngineContext) -> EngineResponse:
        """Produce a single, complete response."""

    @abstractmethod
    def stream(self, ctx: EngineContext) -> AsyncIterator[str]:
        """Yield text chunks as they are generated."""



    @staticmethod
    def parse_recommendations(
        text: str, id_to_title: dict[str, str]
    ) -> list[Recommendation]:
        """Extract <REC>...</REC> item ids from model output and resolve titles."""
        recs: list[Recommendation] = []
        seen: set[str] = set()
        for match in _REC_BLOCK.finditer(text):
            ids_raw = match.group(1)
            for item_id in (x.strip() for x in ids_raw.split(",")):
                if not item_id or item_id in seen:
                    continue
                seen.add(item_id)
                title = id_to_title.get(item_id, "")
                if title:
                    recs.append(Recommendation(item_id=item_id, title=title))
        return recs

    @staticmethod
    def strip_rec_block(text: str) -> str:
        """Remove the <REC> block from user-facing text."""
        return _REC_BLOCK.sub("", text).strip()

    @staticmethod
    def strip_thinking(text: str) -> str:
        """Remove <thinking>...</thinking> blocks from streamed text."""
        return re.sub(
            r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
