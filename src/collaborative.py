"""Bounded item-to-item collaborative retrieval from pre-cutoff behavior."""

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from src.data import (
    DEFAULT_DB, EVENT_WEIGHTS, as_utc, load_events, load_impressions, load_items,
    visitor_history,
)
from src.evaluate import evaluate_snapshot
from src.popularity import recommend as popularity_recommend
from src.split import DEFAULT_OBSERVATION_END, build_snapshots, eligible_items


DEFAULT_ARTIFACT = Path("artifacts/collaborative_validation.json")
RESULT_COLUMNS = ["item_id", "score", "source"]
TRACE_COLUMNS = ["source_item_id", "interaction_strength", "similarity", "contribution"]


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _strengths(events):
    weights = events.event_type.map(EVENT_WEIGHTS)
    if weights.isna().any():
        raise ValueError("Unknown event type")
    return events.assign(strength=weights).groupby(
        ["visitor_id", "item_id"], sort=True).strength.sum()


@dataclass
class CollaborativeModel:
    as_of_time: pd.Timestamp
    neighbors_per_item: int
    item_ids: tuple
    neighbors: dict
    popularity_scores: dict
    visitor_ids: tuple = ()
    matrix: object = field(default=None, repr=False)

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
        if not set(history.item_id) <= set(self.item_ids):
            raise ValueError("history contains unknown or unavailable items")
        strengths = _strengths(history)
        return {item: float(value) for (_, item), value in strengths.items()}, set(
            history.loc[history.event_type == "purchase", "item_id"])

    def generate(self, visitor_id, history, limit):
        """Return pure positive-score candidates, or popularity for empty history."""
        _positive_integer(limit, "limit")
        strengths, purchased = self._history(visitor_id, history)
        if history.empty:
            scores, source = self.popularity_scores, "popularity"
        else:
            scores, source = {}, "collaborative"
            for item, strength in strengths.items():
                for neighbor, similarity in self.neighbors[item]:
                    scores[neighbor] = scores.get(neighbor, 0.0) + strength * similarity
        ranked = sorted(((item, score) for item, score in scores.items()
                         if item not in purchased and (source == "popularity" or score > 0)),
                        key=lambda row: (-row[1], row[0]))[:limit]
        return pd.DataFrame([(item, score, source) for item, score in ranked],
                            columns=RESULT_COLUMNS)

    def explain(self, visitor_id, history, item_id):
        """Trace every retained source contribution to an eligible candidate."""
        strengths, purchased = self._history(visitor_id, history)
        if item_id not in self.neighbors:
            raise ValueError("Unknown or unavailable candidate item")
        rows = []
        if item_id not in purchased:
            for source, strength in strengths.items():
                for neighbor, similarity in self.neighbors[source]:
                    if neighbor == item_id:
                        rows.append((source, strength, similarity, strength * similarity))
        return pd.DataFrame(sorted(rows, key=lambda row: (-row[3], row[0])),
                            columns=TRACE_COLUMNS)

    def save(self, path):
        """Atomically persist only bounded neighbors and serving metadata."""
        payload = {
            "schema_version": 1, "as_of_time": self.as_of_time.isoformat(),
            "event_weights": EVENT_WEIGHTS, "neighbors_per_item": self.neighbors_per_item,
            "item_ids": list(self.item_ids), "neighbors": self.neighbors,
            "popularity_scores": self.popularity_scores,
        }
        _from_payload(payload)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=path.name + ".", suffix=".tmp",
                                             delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path):
        return _from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def _from_payload(payload):
    try:
        if not isinstance(payload, dict):
            raise ValueError("Artifact must be an object")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("Unsupported artifact schema version")
        if (payload["event_weights"] != EVENT_WEIGHTS or
                any(type(value) is not int for value in payload["event_weights"].values())):
            raise ValueError("Artifact event weights do not match this implementation")
        cutoff = as_utc(payload["as_of_time"])
        bound = payload["neighbors_per_item"]
        _positive_integer(bound, "neighbors_per_item")
        if not isinstance(payload["item_ids"], list):
            raise ValueError("Artifact item IDs must be a list")
        items = tuple(payload["item_ids"])
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError("Invalid artifact item IDs")
        if items != tuple(sorted(set(items))):
            raise ValueError("Artifact item IDs must be unique and sorted")
        neighbors, popularity = payload["neighbors"], payload["popularity_scores"]
        if not isinstance(neighbors, dict) or not isinstance(popularity, dict):
            raise ValueError("Artifact mappings must be objects")
        if set(neighbors) != set(items) or set(popularity) != set(items):
            raise ValueError("Artifact catalog mappings do not match")
        for score in popularity.values():
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
                raise ValueError("Invalid artifact popularity score")
        converted = {}
        for item, rows in neighbors.items():
            if not isinstance(rows, (list, tuple)):
                raise ValueError("Artifact neighbors must be lists")
            if len(rows) > bound:
                raise ValueError("Artifact exceeds neighbor bound")
            seen, converted[item] = set(), []
            for neighbor, score in rows:
                if neighbor not in popularity or neighbor == item or neighbor in seen:
                    raise ValueError("Invalid artifact neighbor")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 < score <= 1:
                    raise ValueError("Invalid artifact similarity score")
                seen.add(neighbor)
                converted[item].append((neighbor, float(score)))
            if converted[item] != sorted(converted[item], key=lambda row: (-row[1], row[0])):
                raise ValueError("Artifact neighbors must be ordered")
        return CollaborativeModel(cutoff, bound, items, converted, popularity)
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed collaborative artifact") from exc


def fit_collaborative(events, items, as_of_time, neighbors_per_item=20):
    """Fit on earlier events only; each sparse similarity row is pruned immediately."""
    _positive_integer(neighbors_per_item, "neighbors_per_item")
    cutoff = as_utc(as_of_time)
    item_ids = tuple(sorted(eligible_items(items, cutoff).item_id))
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Duplicate catalog item IDs")
    history = events.loc[pd.to_datetime(events.timestamp, utc=True) < cutoff].copy()
    if not set(history.item_id) <= set(item_ids):
        raise ValueError("Training history contains unknown or unavailable items")
    visitor_ids = tuple(sorted(history.visitor_id.unique()))
    item_index = {item: i for i, item in enumerate(item_ids)}
    visitor_index = {visitor: i for i, visitor in enumerate(visitor_ids)}
    strengths = _strengths(history)
    rows = [visitor_index[visitor] for visitor, _ in strengths.index]
    columns = [item_index[item] for _, item in strengths.index]
    matrix = csr_array((strengths.to_numpy(dtype=np.float64), (rows, columns)),
                       shape=(len(visitor_ids), len(item_ids)), dtype=np.float64)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    item_vectors = matrix.multiply(inverse).T.tocsr()
    neighbors = {}
    for i, item in enumerate(item_ids):
        similarities = (item_vectors[i:i + 1] @ item_vectors.T).tocsr()
        row = [(item_ids[j], float(np.clip(score, 0, 1)))
               for j, score in zip(similarities.indices, similarities.data)
               if j != i and score > 0]
        neighbors[item] = sorted(row, key=lambda pair: (-pair[1], pair[0]))[:neighbors_per_item]
    popularity = dict(zip(item_ids, map(float, np.asarray(matrix.sum(axis=0)).ravel())))
    return CollaborativeModel(cutoff, neighbors_per_item, item_ids, neighbors,
                              popularity, visitor_ids, matrix)


def save_if_useful(model, collaborative, popularity, path=DEFAULT_ARTIFACT):
    """Do not touch an existing artifact unless validation retrieval improves."""
    scores = [result["segments"]["personalized"]["candidate_recall_at_50"]
              for result in (collaborative, popularity)]
    if all(score is not None for score in scores) and scores[0] > scores[1]:
        model.save(path)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("recommend", help="Fit once and inspect visitor recommendations")
    demo.add_argument("visitor_ids", nargs="+")
    demo.add_argument("--as-of", required=True)
    demo.add_argument("--k", type=int, default=10)
    evaluation = commands.add_parser("evaluate", help="Compare validation only; conditionally save")
    evaluation.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    for command in (demo, evaluation):
        command.add_argument("--db", type=Path, default=DEFAULT_DB)
        command.add_argument("--neighbors-per-item", type=int, default=20)
    args = parser.parse_args()
    try:
        events, items = load_events(args.db), load_items(args.db)
        if args.command == "recommend":
            model = fit_collaborative(events, items, args.as_of, args.neighbors_per_item)
            output = []
            for visitor in args.visitor_ids:
                history = visitor_history(visitor, args.as_of, args.db)
                candidates = model.generate(visitor, history, args.k)
                trace = (model.explain(visitor, history, candidates.iloc[0].item_id)
                         if not candidates.empty and not history.empty else pd.DataFrame())
                output.append({"visitor_id": visitor, "history_events": len(history),
                               "recommendations": candidates.to_dict("records"),
                               "first_candidate_trace": trace.to_dict("records")})
        else:
            if args.artifact.resolve() == args.db.resolve():
                raise ValueError("artifact path must differ from database path")
            snapshot = next(s for s in build_snapshots() if s.role == "validation")
            end = as_utc(DEFAULT_OBSERVATION_END)
            if any((frame.timestamp >= end).any() for frame in (events, load_impressions(args.db))):
                raise ValueError("Observed activity extends beyond the declared observation end")
            started = perf_counter()
            model = fit_collaborative(events, items, snapshot.as_of_time, args.neighbors_per_item)
            fit_ms = (perf_counter() - started) * 1000
            popularity = evaluate_snapshot(snapshot, events, items,
                lambda visitor, history, limit: popularity_recommend(
                    visitor, limit, snapshot.as_of_time, args.db))
            collaborative = evaluate_snapshot(snapshot, events, items, model.generate)
            differences = {}
            for segment in collaborative["segments"]:
                differences[segment] = {
                    metric: None if value is None or popularity["segments"][segment][metric] is None
                    else value - popularity["segments"][segment][metric]
                    for metric, value in collaborative["segments"][segment].items()
                    if metric not in ("visitors", "evaluated_visitors", "excluded_no_relevant_outcomes")}
            saved = save_if_useful(model, collaborative, popularity, args.artifact)
            output = {"fit_ms": fit_ms, "neighbors_per_item": args.neighbors_per_item,
                      "popularity": popularity, "collaborative": collaborative,
                      "differences_collaborative_minus_popularity": differences,
                      "artifact_saved": saved, "artifact": str(args.artifact) if saved else None,
                      "artifact_decision": "Saved: validation retrieval improved" if saved else
                      "Not saved: validation retrieval did not strictly improve; existing artifact untouched"}
    except (ValueError, OSError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        parser.exit(1, f"Collaborative command failed: {exc}\n")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
