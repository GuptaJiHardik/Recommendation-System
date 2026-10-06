"""Evaluate recommendation retrieval and ranking on chronological snapshots."""

import argparse
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import pandas as pd

from src.data import DEFAULT_DB, as_utc, load_events, load_impressions, load_items
from src.popularity import recommend
from src.split import (
    DEFAULT_OBSERVATION_END, DEFAULT_SNAPSHOTS, build_snapshots,
    eligible_items, split_at,
)


def _ranked_items(ranked, k):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    ranked = list(ranked)
    if any(not isinstance(item, str) or not item.strip() for item in ranked):
        raise ValueError("recommendation IDs must be nonempty strings")
    if len(ranked) != len(set(ranked)):
        raise ValueError("duplicate recommendation IDs")
    return ranked[:k]


def precision_at_k(ranked, relevant, k=10):
    """Short lists retain the fixed k denominator."""
    ranked, relevant = _ranked_items(ranked, k), set(relevant)
    return sum(item in relevant for item in ranked) / k


def recall_at_k(ranked, relevant, k=10):
    ranked, relevant = _ranked_items(ranked, k), set(relevant)
    return sum(item in relevant for item in ranked) / len(relevant) if relevant else 0.0


def average_precision_at_k(ranked, relevant, k=10):
    """Normalize the sum of precisions at hits by min(k, relevant count)."""
    ranked, relevant = _ranked_items(ranked, k), set(relevant)
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for position, item in enumerate(ranked, 1):
        if item in relevant:
            hits += 1
            total += hits / position
    return total / min(k, len(relevant))


def _candidate_ids(frame, catalog, purchased):
    if not isinstance(frame, pd.DataFrame) or "item_id" not in frame:
        raise ValueError("recommender must return a DataFrame with item_id")
    ids = _ranked_items(frame.item_id.tolist(), 50)
    if len(frame) > 50:
        raise ValueError("recommender returned more than 50 candidates")
    if not set(ids) <= catalog:
        raise ValueError("recommender returned unavailable or unknown items")
    if set(ids) & purchased:
        raise ValueError("recommender returned previously purchased items")
    return ids


def evaluate_snapshot(snapshot, events, items, recommend_fn):
    """recommend_fn(visitor_id, pre-cutoff history, limit) returns ordered items.

    Only visitors with recommendable positive outcomes enter metric averages.
    No future outcomes are passed to the recommender. Latency includes the
    callback and top-10 selection, excluding labels, validation, and metrics.
    """
    history, outcomes = split_at(events, snapshot)
    catalog = set(eligible_items(items, snapshot.as_of_time).item_id)
    histories = {visitor: frame.copy() for visitor, frame in history.groupby("visitor_id")}
    purchases = {
        visitor: set(frame.loc[frame.event_type == "purchase", "item_id"])
        for visitor, frame in histories.items()
    }
    positives = outcomes.loc[
        outcomes.event_type.isin(["add_to_cart", "purchase"]), ["visitor_id", "item_id"]
    ].drop_duplicates()
    relevant = {}
    unavailable, purchased = 0, 0
    for visitor, item in positives.itertuples(index=False, name=None):
        if item not in catalog:
            unavailable += 1
        elif item in purchases.get(visitor, set()):
            purchased += 1
        else:
            relevant.setdefault(visitor, set()).add(item)

    known = set(histories)
    cold = set(outcomes.visitor_id) - known
    rows, segments = [], {}
    for segment, visitors in (("personalized", known), ("cold_start", cold)):
        evaluated = sorted(visitors & set(relevant))
        if evaluated:
            visitor = evaluated[0]
            warmup = recommend_fn(visitor, histories.get(visitor, history.iloc[:0]).copy(), 50)
            _candidate_ids(warmup, catalog, purchases.get(visitor, set()))
        segment_rows = []
        for visitor in evaluated:
            visitor_history = histories.get(visitor, history.iloc[:0]).copy()
            started = perf_counter()
            candidates = recommend_fn(visitor, visitor_history, 50)
            # Selection is timed; output validation and metric work are not.
            top10 = candidates.head(10) if isinstance(candidates, pd.DataFrame) else candidates
            elapsed_ms = (perf_counter() - started) * 1000
            ids = _candidate_ids(candidates, catalog, purchases.get(visitor, set()))
            ranked = top10.item_id.tolist()
            target = relevant[visitor]
            row = {
                "visitor_id": visitor, "segment": segment,
                "relevant_items": sorted(target), "candidate_items": ids,
                "recommendations": ranked,
                "precision_at_10": precision_at_k(ranked, target),
                "recall_at_10": recall_at_k(ranked, target),
                "ap_at_10": average_precision_at_k(ranked, target),
                "candidate_recall_at_50": recall_at_k(ids, target, 50),
                "latency_ms": elapsed_ms,
            }
            segment_rows.append(row)
        summary = {
            "visitors": len(visitors), "evaluated_visitors": len(evaluated),
            "excluded_no_relevant_outcomes": len(visitors) - len(evaluated),
            "precision_at_10": None, "recall_at_10": None, "map_at_10": None,
            "candidate_recall_at_50": None, "catalog_coverage": None,
            "median_latency_ms": None, "p95_latency_ms": None,
        }
        if segment_rows:
            frame = pd.DataFrame(segment_rows)
            for name, column in (("precision_at_10", "precision_at_10"),
                                 ("recall_at_10", "recall_at_10"),
                                 ("map_at_10", "ap_at_10"),
                                 ("candidate_recall_at_50", "candidate_recall_at_50")):
                summary[name] = float(frame[column].mean())
            shown = {item for row in segment_rows for item in row["recommendations"]}
            summary["catalog_coverage"] = len(shown) / len(catalog)
            summary["median_latency_ms"] = float(frame.latency_ms.quantile(0.5))
            summary["p95_latency_ms"] = float(frame.latency_ms.quantile(0.95))
        segments[segment] = summary
        rows.extend(segment_rows)
    return {
        "role": snapshot.role, "snapshot": snapshot.as_of_time.isoformat(),
        "outcome_end": snapshot.outcome_end.isoformat(),
        "counts": {"history_events": len(history), "outcome_events": len(outcomes),
                   "eligible_items": len(catalog), "raw_positive_pairs": len(positives),
                   "excluded_unavailable_pairs": unavailable,
                   "excluded_prior_purchase_pairs": purchased,
                   "relevant_pairs": sum(map(len, relevant.values()))},
        "segments": segments, "visitor_results": rows,
    }


def render_report(results, command, db_path):
    """Build a self-contained Markdown report from measured results."""
    lines = [
        "# Popularity baseline evaluation", "", "## Reproduce", "",
        f"Database: `{db_path}`. Run from the repository root (Python 3.11+).",
        "The database must already be imported with the Phase 1 command.", "",
        "```text", command,
        'uv run --no-project --with "pandas>=2.2,<3" --with "pytest>=8,<10" python -B -m pytest tests -q -p no:cacheprovider',
        "```", "", "## Protocol", "",
        "History is strictly before the UTC snapshot; outcomes are in [snapshot, snapshot + 7 days). "
        "Relevance is a distinct add-to-cart or purchase on an item available at the snapshot, excluding prior purchases. "
        "Views are not relevant outcomes. Missing item creation times mean no known introduction cutoff.", "",
        "Personalized evaluation requires prior event history and at least one relevant item. "
        "Cold start has no prior event history and is reported separately using popularity fallback. "
        "Visitors without relevant outcomes are excluded from metric averages; cold-start visitor counts cover "
        "visitors observed in outcome events. Visitors never observed in events cannot be enumerated.", "",
        "Retrieval considers the full available catalog, without sampled distractors. "
        "Popularity uses pre-snapshot weights view=1, add-to-cart=3, purchase=5, with item-ID ties and "
        "eligible zero-score items filling short lists. Request up to 50 candidates; retain their first 10 as recommendations.", "",
        "Precision@10 = hits / 10, including short lists. Recall@10 = hits / relevant item count. "
        "AP@10 = sum of precision at each hit / min(10, relevant item count). MAP@10 is mean AP. "
        "Candidate Recall@50 uses the same relevance set. All averages weight visitors equally. "
        "Coverage = distinct top-10 items / available catalog size, separately for each segment and snapshot.", "",
        "Latency uses one untimed warm-up per nonempty segment, then one measured call per evaluated visitor. "
        "It includes the popularity callback (SQL retrieval, purchase filtering, ordering, and DataFrame construction) "
        "and top-10 selection. It excludes initial database loads, cohort/history preparation, output validation, "
        "labels, metrics, and report writing. Percentiles use Pandas linear interpolation; this is local batch "
        "recommendation latency, not HTTP request latency. Quality results are deterministic for a fixed database; latency varies.", "",
        "Validation is for comparisons and tuning. Test remains reserved for the final comparison; "
        "running this command does not evaluate other roles. No popularity weights were tuned in Phase 2.",
    ]
    for result in results:
        lines += ["", f"## {result['role'].title()}: {result['snapshot']}", "",
                  f"Outcome end (exclusive): {result['outcome_end']}.", "",
                  "| Dataset count | Value |", "| --- | ---: |"]
        lines += [f"| {name.replace('_', ' ')} | {value} |" for name, value in result["counts"].items()]
        lines += ["", "| Measure | Personalized | Cold start |", "| --- | ---: | ---: |"]
        labels = {
            "visitors": "Observed visitors", "evaluated_visitors": "Evaluated visitors",
            "excluded_no_relevant_outcomes": "Excluded: no relevant outcomes",
            "precision_at_10": "Precision@10", "recall_at_10": "Recall@10",
            "map_at_10": "MAP@10", "candidate_recall_at_50": "Candidate Recall@50",
            "catalog_coverage": "Catalog coverage", "median_latency_ms": "Median latency (ms)",
            "p95_latency_ms": "p95 latency (ms)",
        }
        for name in result["segments"]["personalized"]:
            values = []
            for segment in ("personalized", "cold_start"):
                value = result["segments"][segment][name]
                values.append("not available" if value is None else
                              str(value) if isinstance(value, int) else f"{value:.6f}")
            lines.append(f"| {labels[name]} | {' | '.join(values)} |")
        measured = result["segments"]["personalized"]
        if measured["evaluated_visitors"]:
            lines += ["", f"Observation: across {measured['evaluated_visitors']} evaluated visitors, "
                      f"candidate Recall@50 is {measured['candidate_recall_at_50']:.2%}, "
                      f"but top-10 recall is {measured['recall_at_10']:.2%}; the top-10 lists "
                      f"cover {measured['catalog_coverage']:.2%} of the available catalog."]
        if result["visitor_results"]:
            row = next((row for row in result["visitor_results"] if row["precision_at_10"] > 0),
                       result["visitor_results"][0])
            hits = [i for i, item in enumerate(row["recommendations"], 1) if item in row["relevant_items"]]
            lines += ["", f"### Inspect visitor {row['visitor_id']} ({row['segment']})", "",
                      f"Relevant items: {', '.join(row['relevant_items'])}.", "",
                      f"Top 10 in order: {', '.join(row['recommendations']) or '(empty)' }.", "",
                      f"Hit positions: {hits}. Precision@10 = {len(hits)}/10 = {row['precision_at_10']:.6f}; "
                      f"Recall@10 = {len(hits)}/{len(row['relevant_items'])} = {row['recall_at_10']:.6f}.", "",
                      f"AP@10 = sum(j / hit_position_j) / min(10, {len(row['relevant_items'])}) "
                      f"= {row['ap_at_10']:.6f}. Candidate Recall@50 = {row['candidate_recall_at_50']:.6f}."]
    lines += ["", "## Observation and learning checkpoint", "",
              "Compare candidate Recall@50 with Recall@10 above: their difference measures relevant items "
              "retrieved but lost from the top 10. A future ranker cannot recover relevant items absent from its pool. "
              "An empty cold-start segment provides no measured evidence of fallback quality; tests verify its mechanics.", "",
              "Precision measures how much of the recommendation list is relevant; recall measures how much "
              "of the relevant set is found. MAP rewards placing relevant items early, since each hit contributes "
              "the precision at its rank.", "",
              "Limitation: these are small synthetic, observational outcome windows, evaluated only for visitors "
              "with recommendable positive outcomes. They do not measure all-visitor performance, prove dislike "
              "for missing events, or establish causal business uplift. Exposure and simulated behavior affect outcomes. "
              "A controlled online experiment would be needed to measure business impact. Implicit events are "
              "not numeric ratings, so RMSE is not applicable here.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--role", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--report", type=Path, default=Path("reports/baseline.md"))
    parser.add_argument("--snapshots", nargs="+", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--observation-end", default=DEFAULT_OBSERVATION_END)
    args = parser.parse_args()
    try:
        snapshots = build_snapshots(args.snapshots, args.observation_end)
        events, items, impressions = load_events(args.db), load_items(args.db), load_impressions(args.db)
        end = as_utc(args.observation_end)
        if any((frame.timestamp >= end).any() for frame in (events, impressions)):
            raise ValueError("Observed activity extends beyond the declared observation end")
        if args.report.resolve() == args.db.resolve():
            raise ValueError("report path must differ from database path")
        results = []
        for snapshot in snapshots:
            if snapshot.role == args.role:
                def popularity(visitor_id, history, limit):
                    return recommend(visitor_id, limit, snapshot.as_of_time, args.db)
                results.append(evaluate_snapshot(snapshot, events, items, popularity))
        def quote(value):
            # Reproduction commands target the project's PowerShell environment.
            return "'" + str(value).replace("'", "''") + "'"

        command = ('uv run --no-project --with "pandas>=2.2,<3" python -B -m src.evaluate '
                   f'--db {quote(args.db.as_posix())} --role {args.role} '
                   f'--report {quote(args.report.as_posix())} --snapshots '
                   + " ".join(quote(value) for value in args.snapshots)
                   + f" --observation-end {quote(args.observation_end)}")
        report = render_report(results, command, args.db)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    except (ValueError, OSError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(json.dumps({"report": str(args.report), "snapshots": [
        {key: value for key, value in result.items() if key != "visitor_results"}
        for result in results]}, indent=2))


if __name__ == "__main__":
    main()
