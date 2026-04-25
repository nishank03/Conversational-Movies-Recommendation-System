"""Pydantic models shared across the API, engines, and retrieval layers."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field




class Message(BaseModel):
    """A single turn in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Incoming request to the /chat/stream endpoint."""

    user_id: str | None = Field(
        default=None,
        description=(
            "Optional known user id. If provided, the engine can pull "
            "history_interaction to personalise recommendations."
        ),
    )
    history: list[Message] = Field(
        default_factory=list,
        description="Prior conversation turns (oldest first).",
    )
    message: str = Field(..., min_length=1, description="The new user message.")
    engine: Literal["few_shot", "rag", "agent"] | None = Field(
        default=None,
        description="Override the configured default engine.",
    )
    top_k: int | None = Field(default=None, ge=1, le=50)



class Movie(BaseModel):
    """A movie in the catalog."""

    item_id: str
    title: str
    description: str | None = None
    genre: str | None = None

    def to_context_line(self) -> str:
        """Compact single-line representation for prompt context."""
        parts = [f"- {self.title} (id={self.item_id})"]
        if self.genre:
            parts.append(f"[{self.genre}]")
        if self.description:
            parts.append(f"— {self.description[:160]}")
        return " ".join(parts)


class UserProfile(BaseModel):
    """A user's known preferences, assembled from history_interaction."""

    user_id: str
    history: list[Movie] = Field(default_factory=list)
    might_like: list[Movie] = Field(default_factory=list)


class RetrievedCandidate(BaseModel):
    """A candidate surfaced by the retriever, with its score."""

    movie: Movie
    score: float



class Recommendation(BaseModel):
    """A single movie recommendation with reasoning."""

    item_id: str
    title: str
    reason: str | None = None


class EngineResponse(BaseModel):
    """Structured final response from an engine (non-streaming path)."""

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    engine: str
    prompt_version: str
    latency_ms: float | None = None



class StreamEvent(BaseModel):
    """A single Server-Sent-Event payload."""

    event: Literal["token", "recommendation", "done", "error"]
    data: str
