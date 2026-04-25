"""Evaluation runner.

Replays held-out conversations against an engine and computes ranking metrics 
(Hit@K, Recall@K, etc) by comparing predictions to the ground truth.
Runs concurrently with a bounded semaphore.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crs.config import Settings, get_settings
from crs.crs_engines.base import BaseCRS, EngineContext
from crs.data.loaders import DatasetLoader, get_default_loader
from crs.evaluation.metrics import aggregate, compute_all
from crs.schemas import Message, UserProfile
from crs.utils.logging import get_logger
from crs.utils.timing import timer

logger = get_logger(__name__)


@dataclass
class EvalSampleResult:
    """Metrics + artifacts for a single evaluation sample."""

    user_id: str
    conversation_id: int
    predicted_ids: list[str]
    ground_truth_ids: list[str]
    metrics: dict[str, float]
    latency_ms: float
    reply_preview: str = ""


@dataclass
class EvalReport:
    """Aggregated evaluation report for one engine run."""

    engine: str
    prompt_version: str
    n_samples: int
    aggregate_metrics: dict[str, float]
    per_sample: list[EvalSampleResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "prompt_version": self.prompt_version,
            "n_samples": self.n_samples,
            "aggregate_metrics": self.aggregate_metrics,
            "per_sample": [
                {
                    "user_id": s.user_id,
                    "conversation_id": s.conversation_id,
                    "predicted_ids": s.predicted_ids,
                    "ground_truth_ids": s.ground_truth_ids,
                    "metrics": s.metrics,
                    "latency_ms": s.latency_ms,
                    "reply_preview": s.reply_preview,
                }
                for s in self.per_sample
            ],
        }


class EvaluationRunner:
    """Runs an engine over the eval split and produces an EvalReport."""

    def __init__(
        self,
        engine: BaseCRS,
        loader: DatasetLoader | None = None,
        settings: Settings | None = None,
        concurrency: int = 4,
    ) -> None:
        self.engine = engine
        self.loader = loader or get_default_loader()
        self.settings = settings or get_settings()
        self.concurrency = concurrency

    # ---------- Context reconstruction ----------

    def _build_context_from_record(
        self, record: dict[str, Any]
    ) -> EngineContext | None:
        """Turn a flattened conversation record into an EngineContext.

        The dialogue alternates User:/Agent: turns. We use all turns up to
        (but not including) the FINAL agent turn as history, and the last
        user turn as the current message. The final agent turn contains the
        recommendation we will score against.
        """
        dialogue = record.get("dialogue")
        if not dialogue:
            return None

        turns = _parse_dialogue(dialogue)
        if len(turns) < 2:
            return None

        # Walk backwards; find the last user turn whose next turn is an agent
        # recommendation. Everything before it becomes history.
        last_user_idx = None
        for i in range(len(turns) - 1, -1, -1):
            if turns[i].role == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return None

        history = turns[:last_user_idx]
        current_message = turns[last_user_idx].content

        profile = self.loader.get_user_profile(record["user_id"]) or UserProfile(
            user_id=record["user_id"]
        )
        return EngineContext(
            message=current_message,
            history=history,
            profile=profile,
        )

    # ---------- Single-sample evaluation ----------

    async def _evaluate_one(
        self, record: dict[str, Any], sem: asyncio.Semaphore
    ) -> EvalSampleResult | None:
        async with sem:
            ctx = self._build_context_from_record(record)
            if ctx is None:
                return None

            gt_ids = list(
                dict.fromkeys(
                    (record.get("rec_item") or [])
                    + (record.get("user_might_like") or [])
                )
            )

            try:
                with timer() as t:
                    response = await self.engine.recommend(ctx)
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "Engine failed on user=%s conv=%s: %s",
                    record["user_id"],
                    record["conversation_id"],
                    e,
                )
                return None

            predicted_ids = [r.item_id for r in response.recommendations]
            metrics = compute_all(
                predicted_ids,
                set(gt_ids),
                k_values=self.settings.eval_k_values,
            )

            return EvalSampleResult(
                user_id=record["user_id"],
                conversation_id=record["conversation_id"],
                predicted_ids=predicted_ids,
                ground_truth_ids=gt_ids,
                metrics=metrics,
                latency_ms=response.latency_ms or t["elapsed_ms"],
                reply_preview=response.reply[:200],
            )

    # ---------- Public API ----------

    async def run(
        self,
        eval_records: list[dict[str, Any]],
        limit: int | None = None,
    ) -> EvalReport:
        records = eval_records[:limit] if limit else eval_records
        sem = asyncio.Semaphore(self.concurrency)

        logger.info(
            "Evaluating engine=%s prompt=%s on %d samples (concurrency=%d)",
            self.engine.name,
            self.settings.prompt_version,
            len(records),
            self.concurrency,
        )

        results = await asyncio.gather(
            *(self._evaluate_one(r, sem) for r in records)
        )
        successful = [r for r in results if r is not None]

        per_sample_metrics = [r.metrics for r in successful]
        agg = aggregate(per_sample_metrics)

        return EvalReport(
            engine=self.engine.name,
            prompt_version=self.settings.prompt_version,
            n_samples=len(successful),
            aggregate_metrics=agg,
            per_sample=successful,
        )

    def save_report(self, report: EvalReport, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Saved eval report to %s", path)
        return path


# ---------- Dialogue parser ----------


def _parse_dialogue(dialogue: str) -> list[Message]:
    """Parse 'User: ... \\n\\nAgent: ...' blocks into Message objects."""
    messages: list[Message] = []
    current_role: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_role is not None and buffer:
            content = "\n".join(buffer).strip()
            if content:
                messages.append(Message(role=current_role, content=content))

    for line in dialogue.splitlines():
        stripped = line.strip()
        if stripped.startswith("User:"):
            flush()
            current_role = "user"
            buffer = [stripped[len("User:"):].strip()]
        elif stripped.startswith("Agent:"):
            flush()
            current_role = "assistant"
            buffer = [stripped[len("Agent:"):].strip()]
        else:
            buffer.append(line)

    flush()
    return messages
