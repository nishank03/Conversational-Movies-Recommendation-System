"""SSE streaming helpers for the chat endpoint.

Streams token chunks, strips <thinking> blocks, and emits a
final 'recommendation' event with parsed <REC> item ids.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from crs.crs_engines.base import BaseCRS, EngineContext
from crs.data.loaders import DatasetLoader
from crs.utils.logging import get_logger

logger = get_logger(__name__)


def _sse(event: str, data: str) -> bytes:
    """Format a single SSE frame. Data is newline-escaped per spec."""
    safe = data.replace("\r", "").replace("\n", "\\n")
    return f"event: {event}\ndata: {safe}\n\n".encode("utf-8")


async def stream_engine_response(
    engine: BaseCRS,
    ctx: EngineContext,
    loader: DatasetLoader,
) -> AsyncIterator[bytes]:
    """Consume the engine's token stream and yield SSE frames.

    Side effects:
      - Accumulates the full reply in memory so we can parse <REC> blocks
        at the end and emit a ``recommendation`` event.
      - Strips <thinking> blocks from the text streamed to the client.
    """
    buffer: list[str] = []
    in_thinking = False
    carry = ""  # leftover text we haven't classified yet

    try:
        async for chunk in engine.stream(ctx):
            buffer.append(chunk)
            # Thinking-block stripping across chunk boundaries
            text = carry + chunk
            out_parts: list[str] = []
            while text:
                if in_thinking:
                    end = text.find("</thinking>")
                    if end == -1:
                        carry = ""
                        text = ""
                    else:
                        in_thinking = False
                        text = text[end + len("</thinking>"):]
                else:
                    start = text.find("<thinking>")
                    if start == -1:
                        # Hold back the last few chars so we don't split a tag
                        keep = min(len("<thinking>"), len(text))
                        out_parts.append(text[:-keep] if keep else text)
                        carry = text[-keep:] if keep else ""
                        text = ""
                    else:
                        out_parts.append(text[:start])
                        in_thinking = True
                        text = text[start + len("<thinking>"):]

            clean = "".join(out_parts)
            # Strip any <REC>...</REC> tokens from live stream too
            if clean:
                yield _sse("token", clean)

        # Flush any remaining carry
        if carry and not in_thinking:
            yield _sse("token", carry)

        # Parse recommendations from the full raw buffer
        full = "".join(buffer)
        recs = engine.parse_recommendations(full, loader.item_map)
        payload = json.dumps([r.model_dump() for r in recs], ensure_ascii=False)
        yield _sse("recommendation", payload)

    except Exception as e:  # noqa: BLE001
        logger.exception("Stream failed: %s", e)
        yield _sse("error", json.dumps({"message": str(e)}))

    finally:
        yield _sse("done", "")
