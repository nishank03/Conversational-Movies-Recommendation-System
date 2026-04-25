"""Tool-calling loop for the agent engine.

Alternates between asking the LLM and executing any tool_use blocks it emits.
Bounded by max_iterations so it can't loop forever.
"""
from __future__ import annotations

import json
from typing import Any

from crs.agents.tools import AgentToolbox, Tool
from crs.config import Settings, get_settings
from crs.crs_engines.base import EngineContext
from crs.data.loaders import DatasetLoader
from crs.llm.client import LLMClient
from crs.llm.formatters import render_user_profile
from crs.retrieval.vector_store import VectorStore
from crs.utils.logging import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """ReAct-style loop — fully provider-agnostic via LLMClient.complete_with_tools."""

    def __init__(
        self,
        llm: LLMClient,
        loader: DatasetLoader,
        vector_store: VectorStore,
        max_iterations: int = 4,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.loader = loader
        self.toolbox = AgentToolbox(loader=loader, vector_store=vector_store)
        self.max_iterations = max_iterations
        self.settings = settings or get_settings()



    @staticmethod
    def _tool_to_spec(tool: Tool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }



    async def run(self, ctx: EngineContext, system_prompt: str) -> str:
        tools_spec = [self._tool_to_spec(t) for t in self.toolbox.list_tools()]

        # Opening user message prefixes the user's query with profile info.
        profile_block = render_user_profile(ctx.profile)
        seed_user_content = (
            f"{profile_block}\n\n"
            f"Current conversation so far:\n"
            + "\n".join(f"{m.role}: {m.content}" for m in ctx.history)
            + f"\n\nuser: {ctx.message}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": seed_user_content}
        ]

        for iteration in range(self.max_iterations):
            logger.debug("Agent iteration %d", iteration + 1)
            response = await self.llm.complete_with_tools(
                messages=messages,
                system=system_prompt,
                tools=tools_spec,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            )

            # Append assistant message to history (uses normalised content
            # blocks so the next call round-trips correctly regardless of
            # the underlying provider).
            messages.append(
                {"role": "assistant", "content": response.raw_content}
            )

            if response.stop_reason != "tool_use":
                # Final answer — return text
                return response.text

            # Otherwise: execute every tool call, feed results back
            tool_results: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                tool = self.toolbox.get(tc.name)
                if tool is None:
                    result = json.dumps({"error": f"Unknown tool {tc.name}"})
                else:
                    result = await tool.execute(tc.arguments)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        # Budget exhausted — ask the LLM for a best-effort final answer.
        logger.warning("Agent hit max_iterations=%d", self.max_iterations)
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the tool-call budget. Produce your best "
                    "recommendation now using the information you already have."
                ),
            }
        )
        final = await self.llm.complete_with_tools(
            messages=messages,
            system=system_prompt,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        return final.text
