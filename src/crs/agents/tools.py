"""Tools the agent can call (search, history lookup, movie details)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from crs.data.loaders import DatasetLoader
from crs.retrieval.vector_store import VectorStore
from crs.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Tool:
    """A single tool the agent can call."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]

    async def execute(self, arguments: dict[str, Any]) -> str:
        try:
            return await self.handler(arguments)
        except Exception as e:  # noqa: BLE001
            logger.exception("Tool %s failed", self.name)
            return json.dumps({"error": str(e)})


class AgentToolbox:
    """Builds and holds the tools the agent can call."""

    def __init__(
        self, loader: DatasetLoader, vector_store: VectorStore
    ) -> None:
        self.loader = loader
        self.vector_store = vector_store
        self._tools: dict[str, Tool] = {}
        self._register_default_tools()



    def _register_default_tools(self) -> None:
        self._tools["search_movies"] = Tool(
            name="search_movies",
            description=(
                "Search the movie catalog by a natural-language query. "
                "Returns the top matching movies with their item_ids and titles."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, e.g. 'cozy christmas musicals'.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many results to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
            handler=self._search_movies,
        )
        self._tools["get_user_history"] = Tool(
            name="get_user_history",
            description=(
                "Fetch a user's watch history (a list of movie titles they "
                "have previously interacted with)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "max_items": {"type": "integer", "default": 20},
                },
                "required": ["user_id"],
            },
            handler=self._get_user_history,
        )
        self._tools["lookup_movie"] = Tool(
            name="lookup_movie",
            description=(
                "Look up a movie's full metadata by its item_id. Use this to "
                "verify a recommendation before committing."
            ),
            input_schema={
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
            handler=self._lookup_movie,
        )



    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)



    async def _search_movies(self, args: dict[str, Any]) -> str:
        query = args["query"]
        top_k = int(args.get("top_k", 10))
        results = self.vector_store.search(query, top_k=top_k)
        payload = [
            {
                "item_id": c.movie.item_id,
                "title": c.movie.title,
                "score": round(c.score, 4),
                "description": (c.movie.description or "")[:200],
            }
            for c in results
        ]
        return json.dumps({"results": payload}, ensure_ascii=False)

    async def _get_user_history(self, args: dict[str, Any]) -> str:
        user_id = args["user_id"]
        max_items = int(args.get("max_items", 20))
        profile = self.loader.get_user_profile(user_id)
        if profile is None:
            return json.dumps({"error": f"Unknown user_id {user_id}"})
        titles = [m.title for m in profile.history[:max_items]]
        return json.dumps(
            {"user_id": user_id, "history": titles, "total": len(profile.history)},
            ensure_ascii=False,
        )

    async def _lookup_movie(self, args: dict[str, Any]) -> str:
        item_id = args["item_id"]
        movie = self.loader.get_movie(item_id)
        if movie is None:
            return json.dumps({"error": f"Unknown item_id {item_id}"})
        return json.dumps(movie.model_dump(), ensure_ascii=False)
