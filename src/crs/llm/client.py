"""Async LLM client with streaming.

Provider-agnostic interface — implementations for Anthropic and OpenAI.
All methods are async so FastAPI can multiplex many concurrent requests.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator

from crs.config import Settings, get_settings
from crs.schemas import Message
from crs.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool-calling data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCallInfo:
    """A single tool call returned by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CompletionWithTools:
    """Response from an LLM call that may include tool calls.

    ``raw_content`` stores the assistant content blocks in Anthropic-style
    dict format so they can be appended directly to the messages list for
    multi-turn tool loops.
    """

    text: str
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" or "tool_use"
    raw_content: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract async LLM client."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Non-streaming completion returning the full string."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Async iterator of text chunks."""

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionWithTools:
        """Completion with tool-calling support.

        Messages and tools use Anthropic-style block format.  Each client
        translates internally to its provider's native shape.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    """Anthropic SDK implementation."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import AsyncAnthropic

        self.settings = settings
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_s,
        )

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        kwargs: dict = {
            "model": self.settings.llm_model,
            "messages": payload,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "temperature": (
                temperature if temperature is not None else self.settings.llm_temperature
            ),
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        # response.content is a list of content blocks; join the text pieces
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        kwargs: dict = {
            "model": self.settings.llm_model,
            "messages": payload,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "temperature": (
                temperature if temperature is not None else self.settings.llm_temperature
            ),
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text_chunk in stream.text_stream:
                yield text_chunk

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionWithTools:
        kwargs: dict = {
            "model": self.settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "temperature": (
                temperature if temperature is not None
                else self.settings.llm_temperature
            ),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCallInfo] = []
        raw_content: list[dict[str, Any]] = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
                raw_content.append({"type": "text", "text": block.text})
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCallInfo(
                        id=block.id, name=block.name, arguments=block.input
                    )
                )
                raw_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return CompletionWithTools(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            raw_content=raw_content,
        )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    """OpenAI SDK implementation (chat completions)."""

    def __init__(self, settings: Settings) -> None:
        from openai import AsyncOpenAI

        self.settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_s,
        )

    def _build_messages(
        self, messages: list[Message], system: str | None
    ) -> list[dict]:
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            out.append({"role": m.role, "content": m.content})
        return out

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=self._build_messages(messages, system),
            temperature=(
                temperature if temperature is not None else self.settings.llm_temperature
            ),
            max_tokens=max_tokens or self.settings.llm_max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=self._build_messages(messages, system),
            temperature=(
                temperature if temperature is not None else self.settings.llm_temperature
            ),
            max_tokens=max_tokens or self.settings.llm_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    # ---- tool-calling (translates Anthropic-style blocks to OpenAI) ------

    def _translate_messages_for_tools(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
    ) -> list[dict[str, Any]]:
        """Convert Anthropic-style tool_use / tool_result message blocks into
        the OpenAI chat-completions format."""
        oai: list[dict[str, Any]] = []
        if system:
            oai.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg.get("content")

            if isinstance(content, str):
                oai.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                oai.append({"role": role, "content": content})
                continue

            # Structured content blocks (Anthropic shape)
            text_parts: list[str] = []
            oai_tool_calls: list[dict[str, Any]] = []
            tool_result_msgs: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    oai_tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(
                                block.get("input", {})
                            ),
                        },
                    })
                elif btype == "tool_result":
                    tool_result_msgs.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(block.get("content", "")),
                    })

            if role == "assistant":
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = "".join(text_parts)
                if oai_tool_calls:
                    assistant_msg["tool_calls"] = oai_tool_calls
                oai.append(assistant_msg)
            elif role == "user":
                if text_parts:
                    oai.append({
                        "role": "user",
                        "content": "".join(text_parts),
                    })
                oai.extend(tool_result_msgs)

        return oai

    @staticmethod
    def _translate_tool_schemas(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Anthropic tool spec → OpenAI function-calling spec."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> CompletionWithTools:
        oai_messages = self._translate_messages_for_tools(messages, system)

        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": oai_messages,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "temperature": (
                temperature if temperature is not None
                else self.settings.llm_temperature
            ),
        }
        if tools:
            kwargs["tools"] = self._translate_tool_schemas(tools)

        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        text = msg.content or ""
        tool_calls: list[ToolCallInfo] = []
        raw_content: list[dict[str, Any]] = []

        if text:
            raw_content.append({"type": "text", "text": text})

        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments)
            except (TypeError, json.JSONDecodeError):
                args = {}
            tool_calls.append(
                ToolCallInfo(
                    id=tc.id, name=tc.function.name, arguments=args
                )
            )
            raw_content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": args,
            })

        stop_reason = (
            "tool_use" if finish_reason == "tool_calls" else "end_turn"
        )

        return CompletionWithTools(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw_content=raw_content,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        return AnthropicClient(settings)
    if settings.llm_provider == "openai":
        return OpenAIClient(settings)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


@lru_cache(maxsize=1)
def get_default_client() -> LLMClient:
    return build_client()
