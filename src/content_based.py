"""Content retrieval from equally weighted catalog metadata and past interests."""

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize

from src.collaborative import fit_collaborative
from src.data import (
    DEFAULT_DB, EVENT_WEIGHTS, as_utc, load_events, load_impressions, load_items,
    visitor_history,
)
from src.evaluate import evaluate_snapshot
from src.popularity import recommend as popularity_recommend
from src.split import DEFAULT_OBSERVATION_END, build_snapshots, eligible_items


RESULT_COLUMNS = ["item_id", "score", "source"]
TRACE_COLUMNS = ["source_item_id", "interaction_strength", "similarity", "contribution"]


def _strengths(events):
    weights = events.event_type.map(EVENT_WEIGHTS)
    if weights.isna().any():
        raise ValueError("Unknown event type")
    return events.assign(strength=weights).groupby("item_id", sort=True).strength.sum()


@dataclass
class ContentBasedModel:
    as_of_time: pd.Timestamp
    item_ids: tuple
    matrix: object = field(repr=False)
    popularity_scores: dict
    encoders: dict = field(repr=False)
    vectorizer: object = field(repr=False)
    price_edges: np.ndarray = field(repr=False)
    feature_names: tuple
    feature_blocks: dict
    item_index: dict = field(repr=False)

    def _history(self, visitor_id, history):
        if not isinstance(visitor_id, str) or not visitor_id.strip():
            raise ValueError("visitor_id must be nonempty")
        required = {"visitor_id", "item_id", "event_type", "timestamp"}
        if not isinstance(history, pd.DataFrame) or not required <= set(history.columns):
            raise ValueError("history must contain canonical event columns")
        if not history.visitor_id.eq(visitor_id).all():
            raise ValueError("history contains another visitor's events")
        timestamps = pd.to_datetime(history.timestamp, utc=True)
        if timestamps.isna().any() or (timestamps >= self.as_of_time).any():
            raise ValueError("history must be strictly before the model cutoff")
        if not set(history.item_id) <= set(self.item_index):
            raise ValueError("history contains unknown or unavailable items")
        return _strengths(history), set(history.loc[history.event_type == "purchase", "item_id"])

    def _profile(self, strengths):
        """Weighted mean before normalization; only one feature row is assembled."""
        indices = [self.item_index[item] for item in strengths.index]
        weights = strengths.to_numpy(dtype=np.float64)
        profile = csr_matrix(weights.reshape(1, -1)) @ self.matrix[indices]
        profile = profile / weights.sum()
        norm = float(np.sqrt(profile.multiply(profile).sum()))
        return profile, norm

    def generate(self, visitor_id, history, limit):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        strengths, purchased = self._history(visitor_id, history)
        if history.empty:
            scores, source = self.popularity_scores, "popularity"
        else:
            profile, norm = self._profile(strengths)
            values = ((self.matrix @ (profile / norm).T).toarray().ravel()
                      if norm else np.zeros(len(self.item_ids)))
            scores = dict(zip(self.item_ids, map(float, np.clip(values, 0, 1))))
            source = "content"
        ranked = sorted(((item, score) for item, score in scores.items()
                         if item not in purchased and (source == "popularity" or score > 0)),
                        key=lambda row: (-row[1], row[0]))[:limit]
        return pd.DataFrame([(item, score, source) for item, score in ranked],
                            columns=RESULT_COLUMNS)

    def explain(self, visitor_id, history, item_id):
        """Source contributions sum to the cosine score, including profile normalization."""
        strengths, purchased = self._history(visitor_id, history)
        if item_id not in self.item_index:
            raise ValueError("Unknown or unavailable candidate item")
        rows = []
        if not history.empty and item_id not in purchased:
            _, norm = self._profile(strengths)
            target = self.matrix[self.item_index[item_id]]
            if norm:
                for source, strength in strengths.items():
                    similarity = float(np.clip(
                        (self.matrix[self.item_index[source]] @ target.T).toarray()[0, 0], 0, 1))
                    if similarity > 0:
                        contribution = float(strength * similarity / (strengths.sum() * norm))
                        rows.append((source, float(strength), similarity, contribution))
        return pd.DataFrame(sorted(rows, key=lambda row: (-row[3], row[0])),
                            columns=TRACE_COLUMNS)


def fit_content_based(events, items, as_of_time):
    """Fit catalog transformations as of the cutoff; outcomes never build profiles."""
    cutoff = as_utc(as_of_time)
    catalog = eligible_items(items, cutoff).sort_values("item_id").copy()
    required = {"item_id", "category", "brand", "price", "text"}
    if not required <= set(catalog.columns):
        raise ValueError("catalog must contain canonical item columns")
    for column in ("item_id", "category", "brand"):
        if not catalog[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"catalog {column} must be nonempty strings")
    if catalog.item_id.duplicated().any():
        raise ValueError("Duplicate catalog item IDs")
    prices = catalog.price.to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("catalog price must be finite and nonnegative")
    item_ids = tuple(catalog.item_id)
    history = events.loc[pd.to_datetime(events.timestamp, utc=True) < cutoff].copy()
    if not set(history.item_id) <= set(item_ids):
        raise ValueError("Training history contains unknown or unavailable items")
    strengths = _strengths(history)
    popularity = {item: float(strengths.get(item, 0)) for item in item_ids}
    encoders, blocks, names, matrices = {}, {}, [], []
    vectorizer, price_edges = None, np.array([], dtype=np.float64)

    def add_block(name, matrix, features):
        start = len(names)
        names.extend(features)
        blocks[name] = (start, len(names))
        matrices.append(normalize(matrix, norm="l2") if matrix.shape[1] else matrix)

    if item_ids:
        for column in ("category", "brand"):
            encoder = OneHotEncoder(sparse_output=True, dtype=np.float64)
            encoded = encoder.fit_transform(catalog[[column]])
            encoders[column] = encoder
            add_block(column, encoded, encoder.get_feature_names_out([column]))
        price_edges = np.unique(np.quantile(prices, [0, 0.2, 0.4, 0.6, 0.8, 1], method="linear"))
        # Interior ties enter the higher bucket; constant prices have no interior edge.
        buckets = np.searchsorted(price_edges[1:-1], prices, side="right").reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=True, dtype=np.float64)
        encoded = encoder.fit_transform(buckets)
        encoders["price"] = encoder
        add_block("price", encoded, encoder.get_feature_names_out(["price_bucket"]))
        text = catalog.text.fillna("").astype(str)
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), smooth_idf=True,
                                     norm="l2", stop_words=None, dtype=np.float64)
        # Detect an empty vocabulary before fitting, rather than swallowing other errors.
        if any(vectorizer.build_analyzer()(description) for description in text):
            encoded = vectorizer.fit_transform(text)
            add_block("text", encoded, [f"text_{word}" for word in vectorizer.get_feature_names_out()])
        else:
            vectorizer = None
            add_block("text", csr_matrix((len(item_ids), 0)), [])
        matrix = normalize(hstack(matrices, format="csr"), norm="l2").tocsr()
    else:
        matrix = csr_matrix((0, 0), dtype=np.float64)
        blocks = {name: (0, 0) for name in ("category", "brand", "price", "text")}
    return ContentBasedModel(cutoff, item_ids, matrix, popularity, encoders, vectorizer,
                             price_edges, tuple(names), blocks,
                             {item: i for i, item in enumerate(item_ids)})


def _diagnostics(result, model, events, items):
    """Describe category concentration and metadata-only retrieval outside timed calls."""
    catalog = eligible_items(items, model.as_of_time).set_index("item_id")
    history = events.loc[events.timestamp < model.as_of_time]
    unseen = set(catalog.index) - set(history.item_id)
    retrieved, rows = set(), []
    for row in result["visitor_results"]:
        if row["segment"] != "personalized":
            continue
        counts = catalog.loc[row["recommendations"], "category"].value_counts().sort_index()
        new_items = [item for item in row["candidate_items"] if item in unseen]
        retrieved.update(new_items)
        example = None
        if new_items:
            item = new_items[0]
            user = history.loc[history.visitor_id == row["visitor_id"]]
            scores = model.generate(row["visitor_id"], user, 50).set_index("item_id")
            example = {"item_id": item, **catalog.loc[item, ["category", "brand", "price", "text"]].to_dict(),
                       "score": float(scores.loc[item, "score"])}
        rows.append({"visitor_id": row["visitor_id"], "top10_category_counts": counts.to_dict(),
                     "largest_category_share": float(counts.max() / counts.sum()) if len(counts) else None,
                     "zero_history_candidate_items": new_items, "new_item_example": example})
    shares = [row["largest_category_share"] for row in rows if row["largest_category_share"] is not None]
    return {"zero_history_catalog_items": sorted(unseen),
            "zero_history_items_retrieved": sorted(retrieved),
            "median_largest_category_share": float(np.median(shares)) if shares else None,
            "visitor_results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("recommend", help="Fit once and inspect visitor interests")
    demo.add_argument("visitor_ids", nargs="+")
    demo.add_argument("--as-of", required=True)
    demo.add_argument("--k", type=int, default=10)
    evaluation = commands.add_parser("evaluate", help="Compare content, collaborative, and popularity on validation")
    for command in (demo, evaluation):
        command.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    try:
        events, items = load_events(args.db), load_items(args.db)
        if args.command == "recommend":
            model = fit_content_based(events, items, args.as_of)
            output = []
            for visitor in args.visitor_ids:
                history = visitor_history(visitor, args.as_of, args.db)
                candidates = model.generate(visitor, history, args.k)
                trace = (model.explain(visitor, history, candidates.iloc[0].item_id)
                         if not candidates.empty and not history.empty else pd.DataFrame())
                details = candidates.merge(items[["item_id", "category", "brand", "price", "text"]],
                                           on="item_id", how="left", sort=False)
                output.append({"visitor_id": visitor, "history_events": len(history),
                               "recommendations": details.to_dict("records"),
                               "first_candidate_trace": trace.to_dict("records")})
        else:
            snapshot = next(s for s in build_snapshots() if s.role == "validation")
            end = as_utc(DEFAULT_OBSERVATION_END)
            if any((frame.timestamp >= end).any() for frame in (events, load_impressions(args.db))):
                raise ValueError("Observed activity extends beyond the declared observation end")
            started = perf_counter()
            model = fit_content_based(events, items, snapshot.as_of_time)
            content_fit_ms = (perf_counter() - started) * 1000
            started = perf_counter()
            collaborative_model = fit_collaborative(events, items, snapshot.as_of_time)
            collaborative_fit_ms = (perf_counter() - started) * 1000
            popularity = evaluate_snapshot(snapshot, events, items,
                lambda visitor, history, limit: popularity_recommend(visitor, limit, snapshot.as_of_time, args.db))
            collaborative = evaluate_snapshot(snapshot, events, items, collaborative_model.generate)
            content = evaluate_snapshot(snapshot, events, items, model.generate)
            differences = {}
            for name, baseline in (("popularity", popularity), ("collaborative", collaborative)):
                differences[name] = {
                    segment: {metric: None if value is None or baseline["segments"][segment][metric] is None
                              else value - baseline["segments"][segment][metric]
                              for metric, value in summary.items()
                              if metric not in ("visitors", "evaluated_visitors", "excluded_no_relevant_outcomes")}
                    for segment, summary in content["segments"].items()}
            output = {"content_fit_ms": content_fit_ms, "collaborative_fit_ms": collaborative_fit_ms,
                      "feature_blocks": model.feature_blocks, "price_edges": model.price_edges.tolist(),
                      "popularity": popularity, "collaborative": collaborative, "content": content,
                      "differences_content_minus": differences,
                      "content_diagnostics": _diagnostics(content, model, events, items)}
    except (ValueError, OSError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        parser.exit(1, f"Content command failed: {exc}\n")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
