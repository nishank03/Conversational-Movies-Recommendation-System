"""Helpers that convert domain objects into prompt-ready strings/messages."""
from __future__ import annotations

from crs.schemas import Message, Movie, RetrievedCandidate, UserProfile


def history_to_messages(
    history: list[Message], new_user_message: str
) -> list[Message]:
    """Append the new user turn to existing history, filtering system messages."""
    filtered = [m for m in history if m.role in ("user", "assistant")]
    return [*filtered, Message(role="user", content=new_user_message)]


def render_user_profile(
    profile: UserProfile | None, max_items: int = 15
) -> str:
    """Render a UserProfile into a compact bullet list for the system prompt."""
    if profile is None or not profile.history:
        return "No prior watch history available for this user."

    lines = ["The user's watch history (most recent first):"]
    for movie in profile.history[:max_items]:
        lines.append(f"  - {movie.title}")
    if len(profile.history) > max_items:
        lines.append(f"  ... and {len(profile.history) - max_items} more")
    return "\n".join(lines)


def render_candidates(
    candidates: list[RetrievedCandidate], max_items: int = 20
) -> str:
    """Render retrieved candidates into a numbered list for the prompt."""
    if not candidates:
        return "No candidate movies retrieved."

    lines = ["Candidate movies (ranked by semantic relevance):"]
    for i, c in enumerate(candidates[:max_items], start=1):
        desc = c.movie.description or ""
        extra = f" — {desc[:140]}" if desc else ""
        lines.append(f"  {i}. {c.movie.title} (id={c.movie.item_id}){extra}")
    return "\n".join(lines)


def render_dialogue_excerpt(dialogue: str, max_chars: int = 800) -> str:
    """Trim a raw dialogue to a safe length for few-shot prompts."""
    clean = dialogue.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit("\n", 1)[0] + "\n..."


def movies_to_title_list(movies: list[Movie]) -> str:
    return ", ".join(f'"{m.title}"' for m in movies) if movies else "(none)"
