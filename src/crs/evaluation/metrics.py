"""Standard IR ranking metrics for evaluation (Hit@K, Recall@K, MRR, NDCG)."""
from __future__ import annotations

import math
from typing import Iterable


def _truncate(predictions: Iterable[str], k: int) -> list[str]:
    return list(predictions)[:k]


def hit_at_k(predictions: list[str], ground_truth: set[str], k: int) -> float:
    """1.0 if any of the top-K predictions is in the ground truth else 0.0."""
    if not ground_truth:
        return 0.0
    top = _truncate(predictions, k)
    return 1.0 if any(p in ground_truth for p in top) else 0.0


def recall_at_k(predictions: list[str], ground_truth: set[str], k: int) -> float:
    """Fraction of ground-truth items captured in the top-K."""
    if not ground_truth:
        return 0.0
    top = _truncate(predictions, k)
    hits = sum(1 for p in top if p in ground_truth)
    return hits / len(ground_truth)


def mrr_at_k(predictions: list[str], ground_truth: set[str], k: int) -> float:
    """Mean reciprocal rank of the first ground-truth hit within top-K."""
    if not ground_truth:
        return 0.0
    for rank, p in enumerate(_truncate(predictions, k), start=1):
        if p in ground_truth:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predictions: list[str], ground_truth: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (binary gains)."""
    if not ground_truth:
        return 0.0

    top = _truncate(predictions, k)
    dcg = 0.0
    for i, p in enumerate(top, start=1):
        if p in ground_truth:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_all(
    predictions: list[str],
    ground_truth: set[str],
    k_values: Iterable[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute every metric at every requested K in one pass."""
    out: dict[str, float] = {}
    for k in k_values:
        out[f"hit@{k}"] = hit_at_k(predictions, ground_truth, k)
        out[f"recall@{k}"] = recall_at_k(predictions, ground_truth, k)
        out[f"mrr@{k}"] = mrr_at_k(predictions, ground_truth, k)
        out[f"ndcg@{k}"] = ndcg_at_k(predictions, ground_truth, k)
    return out


def aggregate(per_sample: list[dict[str, float]]) -> dict[str, float]:
    """Mean each metric across evaluation samples."""
    if not per_sample:
        return {}
    keys = per_sample[0].keys()
    return {k: sum(s[k] for s in per_sample) / len(per_sample) for k in keys}
