"""Eligibility filtering and a soft category limit after model scoring."""

from collections import Counter
from math import ceil, isfinite

import numpy as np
import pandas as pd


def apply_final_ranking(ranked, catalog, purchased, k, category_share=0.4):
    """Return preferred diverse items, then score-ordered backfill if needed.

    The cap is ceil(category_share * requested k), not a fraction of the pool.
    Backfill may exceed it; model scores are never changed. Catalog may contain
    an item_id column or use item_id as its index (as RankFeatureContext does).
    Pass category_share=None to keep ordinary score order.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if category_share is not None and (
        isinstance(category_share, bool)
        or not isinstance(category_share, (int, float))
        or not isfinite(category_share)
        or not 0 < category_share <= 1
    ):
        raise ValueError("category_share must be in (0, 1] or None")
    if not isinstance(ranked, pd.DataFrame) or not {"item_id", "score"} <= set(ranked):
        raise ValueError("ranked must contain item_id and score")
    if ranked.item_id.duplicated().any():
        raise ValueError("Duplicate ranked item IDs")
    if not np.isfinite(ranked.score.to_numpy(dtype=float)).all():
        raise ValueError("Ranking scores must be finite")
    if not isinstance(catalog, pd.DataFrame) or "category" not in catalog:
        raise ValueError("catalog must contain category")
    indexed = catalog.set_index("item_id") if "item_id" in catalog else catalog
    if not indexed.index.is_unique or not indexed.category.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Catalog must have unique item IDs and nonempty categories")
    eligible = ranked.loc[
        ranked.item_id.isin(indexed.index) & ~ranked.item_id.isin(purchased)
    ].sort_values(["score", "item_id"], ascending=[False, True]).reset_index(drop=True)
    if category_share is None or eligible.empty:
        return eligible.head(k).copy()

    cap = ceil(category_share * k)
    counts = Counter()
    selected, skipped = [], []
    for position, item in enumerate(eligible.item_id):
        category = indexed.at[item, "category"]
        if counts[category] < cap:
            selected.append(position)
            counts[category] += 1
            if len(selected) == k:
                break
        else:
            skipped.append(position)
    selected.extend(skipped[:k - len(selected)])
    return eligible.iloc[selected].reset_index(drop=True).copy()
