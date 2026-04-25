"""Evaluation: metrics, runner, and report generation."""

from crs.evaluation.metrics import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from crs.evaluation.runner import EvaluationRunner

__all__ = [
    "hit_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "EvaluationRunner",
]
