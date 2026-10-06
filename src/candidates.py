"""Bounded, source-preserving candidate union; this module does not rank items."""

import numpy as np
import pandas as pd

from src.data import as_utc


CANDIDATE_BUDGETS = {"collaborative": 20, "content": 20, "popularity": 10}
SOURCE_COLUMNS = {
    "collaborative": ("collab_score", "from_collab"),
    "content": ("content_score", "from_content"),
    "popularity": ("popularity_score", "from_popularity"),
}
CANDIDATE_COLUMNS = ["item_id", *(pair[0] for pair in SOURCE_COLUMNS.values()),
                     *(pair[1] for pair in SOURCE_COLUMNS.values())]


def validate_candidates(candidates):
    """A missing score is zero with flag zero; nominated zero scores keep flag one."""
    if not isinstance(candidates, pd.DataFrame) or not set(CANDIDATE_COLUMNS) <= set(candidates):
        raise ValueError("candidates must contain item_id, source scores, and source flags")
    if not candidates.item_id.map(lambda item: isinstance(item, str) and bool(item.strip())).all():
        raise ValueError("candidate item IDs must be nonempty strings")
    if candidates.item_id.duplicated().any():
        raise ValueError("Duplicate candidate item IDs")
    for score, flag in SOURCE_COLUMNS.values():
        values = candidates[score].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("candidate scores must be finite and nonnegative")
        if not candidates[flag].isin([0, 1]).all():
            raise ValueError("source flags must be binary")
        if ((candidates[flag] == 0) & (candidates[score] != 0)).any():
            raise ValueError("Absent source scores must be zero")
    flags = [pair[1] for pair in SOURCE_COLUMNS.values()]
    if candidates[flags].sum(axis=1).eq(0).any():
        raise ValueError("Every candidate must have a nominating source")


def generate_candidates(visitor_id, history, collaborative_model, content_model):
    """Return an item-ID ordered union of up to 20/20/10 nominations.

    Models are freshly fitted at one shared cutoff. Their fallback outputs are
    never counted as personalized nominations. Empty histories get ten popular
    items; personalized shortfalls are not backfilled beyond these budgets.
    """
    if as_utc(collaborative_model.as_of_time) != as_utc(content_model.as_of_time):
        raise ValueError("Candidate models must use the same cutoff")
    catalog = set(collaborative_model.item_ids)
    if catalog != set(content_model.item_ids):
        raise ValueError("Candidate models must use the same catalog")
    # Both existing public interfaces validate the visitor and pre-cutoff history.
    collab = collaborative_model.generate(visitor_id, history, CANDIDATE_BUDGETS["collaborative"])
    content = content_model.generate(visitor_id, history, CANDIDATE_BUDGETS["content"])
    purchased = set(history.loc[history.event_type.eq("purchase"), "item_id"])
    popular = sorted(((item, float(score)) for item, score in
                      collaborative_model.popularity_scores.items()
                      if item in catalog and item not in purchased),
                     key=lambda pair: (-pair[1], pair[0]))[:CANDIDATE_BUDGETS["popularity"]]
    rows = {}

    def nominate(item, score, source):
        if item not in catalog or item in purchased:
            raise ValueError("Generator nominated an unavailable or previously purchased item")
        row = rows.setdefault(item, dict.fromkeys(CANDIDATE_COLUMNS[1:], 0))
        score_column, flag = SOURCE_COLUMNS[source]
        row[score_column], row[flag] = float(score), 1

    if not history.empty:
        for source, frame in (("collaborative", collab), ("content", content)):
            if len(frame) > CANDIDATE_BUDGETS[source] or frame.item_id.duplicated().any():
                raise ValueError("Generator violated its nomination budget or returned duplicates")
            for row in frame.itertuples(index=False):
                if row.source != source:
                    raise ValueError("Unexpected personalized generator source")
                nominate(row.item_id, row.score, source)
    for item, score in popular:
        nominate(item, score, "popularity")
    result = pd.DataFrame([{"item_id": item, **rows[item]} for item in sorted(rows)],
                          columns=CANDIDATE_COLUMNS)
    for score, flag in SOURCE_COLUMNS.values():
        result[score] = result[score].astype("float64")
        result[flag] = result[flag].astype("int64")
    validate_candidates(result)
    return result
