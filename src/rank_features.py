"""Shared, as-of-time ranker features for batch datasets and later serving."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.candidates import CANDIDATE_COLUMNS, validate_candidates
from src.data import EVENT_WEIGHTS, as_utc
from src.split import eligible_items


FEATURE_COLUMNS = (
    *CANDIDATE_COLUMNS[1:],
    "visitor_view_count", "visitor_cart_count", "visitor_purchase_count",
    "days_since_last_activity", "visitor_category_affinity",
    "item_interaction_count", "item_price", "visitor_item_price_gap",
)


@dataclass(frozen=True)
class RankFeatureContext:
    as_of_time: pd.Timestamp
    catalog: pd.DataFrame = field(repr=False)
    history: pd.DataFrame = field(repr=False)
    histories: dict = field(repr=False)
    visitor_stats: dict = field(repr=False)
    item_counts: dict = field(repr=False)


def prepare_rank_context(events, items, as_of_time):
    """Compute aggregates once using strictly earlier events and available items.

    Catalog attributes are assumed static; created_at supplies availability.
    Full event logs may be passed, but no future event enters this context.
    """
    cutoff = as_utc(as_of_time)
    required = {"visitor_id", "item_id", "event_type", "timestamp"}
    if not isinstance(events, pd.DataFrame) or not required <= set(events):
        raise ValueError("events must contain canonical event columns")
    times = pd.to_datetime(events.timestamp, utc=True)
    if times.isna().any():
        raise ValueError("events must have valid timestamps")
    history = events.loc[times < cutoff].copy()
    history["timestamp"] = times.loc[history.index]
    catalog = eligible_items(items, cutoff).sort_values("item_id").set_index("item_id")
    if not {"category", "price"} <= set(catalog) or not catalog.index.is_unique:
        raise ValueError("catalog must contain unique item IDs, categories, and prices")
    prices = catalog.price.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("catalog prices must be finite and nonnegative")
    if not set(history.item_id) <= set(catalog.index):
        raise ValueError("history contains unknown or unavailable items")
    if not history.event_type.isin(EVENT_WEIGHTS).all():
        raise ValueError("Unknown event type")
    if not history.visitor_id.map(lambda visitor: isinstance(visitor, str) and bool(visitor.strip())).all():
        raise ValueError("visitor IDs must be nonempty strings")
    histories, stats = {}, {}
    for visitor, frame in history.groupby("visitor_id", sort=True):
        histories[visitor] = frame.copy()
        weights = frame.event_type.map(EVENT_WEIGHTS).astype(float)
        total = float(weights.sum())
        categories = frame.item_id.map(catalog.category)
        category_weights = weights.groupby(categories).sum()
        counts = frame.event_type.value_counts()
        stats[visitor] = {
            "visitor_view_count": int(counts.get("view", 0)),
            "visitor_cart_count": int(counts.get("add_to_cart", 0)),
            "visitor_purchase_count": int(counts.get("purchase", 0)),
            "days_since_last_activity": (cutoff - frame.timestamp.max()).total_seconds() / 86400,
            "category_affinities": (category_weights / total).to_dict(),
            "mean_price": float((frame.item_id.map(catalog.price) * weights).sum() / total),
            "purchased": set(frame.loc[frame.event_type.eq("purchase"), "item_id"]),
        }
    return RankFeatureContext(cutoff, catalog, history, histories, stats,
                              history.item_id.value_counts().to_dict())


def build_rank_features(visitor_id, candidates, as_of_time, context):
    """Return item_id and the ordered feature schema, without labels or identifiers.

    item_id is join metadata, never a model input. Missing recency/price-gap for
    an empty history remains NaN for Phase 6's preprocessing pipeline.
    """
    if not isinstance(visitor_id, str) or not visitor_id.strip():
        raise ValueError("visitor_id must be nonempty")
    if as_utc(as_of_time) != context.as_of_time:
        raise ValueError("Feature context must use the requested cutoff")
    validate_candidates(candidates)
    if not set(candidates.item_id) <= set(context.catalog.index):
        raise ValueError("Candidates contain unknown or unavailable items")
    stats = context.visitor_stats.get(visitor_id, {})
    if set(candidates.item_id) & stats.get("purchased", set()):
        raise ValueError("Candidates contain previously purchased items")
    result = candidates.loc[:, CANDIDATE_COLUMNS].copy().reset_index(drop=True)
    for name in ("visitor_view_count", "visitor_cart_count", "visitor_purchase_count"):
        result[name] = stats.get(name, 0)
    result["days_since_last_activity"] = stats.get("days_since_last_activity", np.nan)
    categories = result.item_id.map(context.catalog.category)
    result["visitor_category_affinity"] = categories.map(stats.get("category_affinities", {})).fillna(0.0)
    result["item_interaction_count"] = result.item_id.map(context.item_counts).fillna(0).astype("int64")
    result["item_price"] = result.item_id.map(context.catalog.price).astype(float)
    result["visitor_item_price_gap"] = (result.item_price - stats.get("mean_price", np.nan)).abs()
    return result.loc[:, ["item_id", *FEATURE_COLUMNS]]
