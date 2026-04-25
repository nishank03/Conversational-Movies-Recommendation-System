"""Few-shot example pool for the FewShotCRS engine.

We don't hardcode examples — we sample them from the training split built by
``data.split`` and format them as short dialogue snippets. This keeps
examples representative of the real dataset distribution and lets us swap
in more examples without code changes.
"""
from __future__ import annotations

import random
from functools import lru_cache

from crs.data.loaders import DatasetLoader
from crs.data.split import load_split
from crs.llm.formatters import render_dialogue_excerpt
from crs.utils.logging import get_logger

logger = get_logger(__name__)


def _format_example(record: dict, loader: DatasetLoader) -> str | None:
    """Render one training record as a compact example block."""
    dialogue = record.get("dialogue")
    if not dialogue:
        return None

    likes = [loader.item_map.get(i, i) for i in record.get("user_likes", [])]
    rec_ids = record.get("rec_item", [])
    recs = [loader.item_map.get(i, i) for i in rec_ids]

    if not recs:
        return None

    likes_str = ", ".join(f'"{t}"' for t in likes) or "(none stated)"
    recs_str = ", ".join(f'"{t}"' for t in recs)
    rec_ids_str = ", ".join(rec_ids)

    return (
        f"### Example\n"
        f"User signals: liked {likes_str}\n"
        f"Dialogue:\n{render_dialogue_excerpt(dialogue, max_chars=600)}\n"
        f"Good recommendation: {recs_str}\n"
        f"<REC>{rec_ids_str}</REC>"
    )


@lru_cache(maxsize=4)
def build_few_shot_block(n_examples: int = 3, seed: int = 7) -> str:
    """Build a cached few-shot block string.

    Cached by (n_examples, seed) so eval runs are reproducible.
    """
    try:
        train = load_split("train")
    except FileNotFoundError:
        logger.warning("Train split not found; few-shot block will be empty.")
        return ""

    loader = DatasetLoader()
    rng = random.Random(seed)
    rng.shuffle(train)

    blocks: list[str] = []
    for record in train:
        formatted = _format_example(record, loader)
        if formatted:
            blocks.append(formatted)
        if len(blocks) >= n_examples:
            break

    if not blocks:
        return ""

    return (
        "Here are a few examples of good recommendation conversations "
        "from our dataset. Study them for style and grounding:\n\n"
        + "\n\n".join(blocks)
    )
