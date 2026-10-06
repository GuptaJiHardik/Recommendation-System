"""Build labeled candidate snapshots, sampled training rows, and retrieval analysis."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd

from src.candidates import CANDIDATE_BUDGETS, SOURCE_COLUMNS, generate_candidates
from src.collaborative import fit_collaborative
from src.content_based import fit_content_based
from src.data import DEFAULT_DB, EVENT_WEIGHTS, as_utc, load_events, load_impressions, load_items
from src.evaluate import evaluate_snapshot
from src.rank_features import FEATURE_COLUMNS, build_rank_features, prepare_rank_context
from src.split import DEFAULT_OBSERVATION_END, OUTCOME_LENGTH, build_snapshots, split_at


DEFAULT_OUTPUT = Path("data/rank")
DEFAULT_REPORT = Path("reports/candidate_analysis.md")
ROW_COLUMNS = ["role", "snapshot", "outcome_end", "visitor_id", "item_id", *FEATURE_COLUMNS,
               "label", "exposed_in_window", "negative_provenance"]


def _records(frame):
    # JSON's null represents missing features; identifiers and UTC strings stay intact.
    return json.loads(frame.to_json(orient="records"))


def _trace(row, context, outcomes, exposures):
    visitor, item = row["visitor_id"], row["item_id"]
    history = context.histories[visitor]
    weights = history.event_type.map(EVENT_WEIGHTS)
    category = context.catalog.loc[item, "category"]
    in_category = history.item_id.map(context.catalog.category).eq(category)
    target = outcomes.loc[outcomes.visitor_id.eq(visitor) & outcomes.item_id.eq(item)
                          & outcomes.event_type.isin(["add_to_cart", "purchase"])]
    shown = exposures.loc[exposures.visitor_id.eq(visitor) & exposures.item_id.eq(item)]
    created = context.catalog.loc[item, "created_at"]
    return {
        "row": _records(pd.DataFrame([row]))[0],
        "history_last_timestamp": history.timestamp.max().isoformat(),
        "visitor_weighted_strength": int(weights.sum()),
        "candidate_category": category,
        "category_weighted_strength": int(weights.loc[in_category].sum()),
        "visitor_weighted_mean_price": context.visitor_stats[visitor]["mean_price"],
        "item_created_at": None if pd.isna(created) else created.isoformat(),
        "target_outcome_timestamps": sorted(t.isoformat() for t in target.timestamp),
        "impression_timestamps": sorted(t.isoformat() for t in shown.timestamp),
    }


def build_snapshot_dataset(snapshot, events, items, impressions=None):
    """Return (all labeled rows, diagnostics), fitting generators at this cutoff.

    Visitors come exclusively from past events. Outcomes and future impressions
    attach labels/provenance after feature construction, never model features.
    The final test snapshot is deliberately unavailable in Phase 5.
    """
    if snapshot.role not in ("train", "validation"):
        raise ValueError("Phase 5 only builds training and validation snapshots")
    if snapshot.outcome_end != snapshot.as_of_time + OUTCOME_LENGTH:
        raise ValueError("Labels require a seven-day outcome window")
    context = prepare_rank_context(events, items, snapshot.as_of_time)
    _, outcomes = split_at(events, snapshot)
    if impressions is None:
        exposures = pd.DataFrame(columns=["visitor_id", "item_id", "timestamp"])
    else:
        _, exposures = split_at(impressions, snapshot)
    positive_pairs = set(outcomes.loc[outcomes.event_type.isin(["add_to_cart", "purchase"]),
                                     ["visitor_id", "item_id"]].itertuples(index=False, name=None))
    exposed_pairs = set(exposures[["visitor_id", "item_id"]].itertuples(index=False, name=None))
    collaborative = fit_collaborative(context.history, items, snapshot.as_of_time)
    content = fit_content_based(context.history, items, snapshot.as_of_time)
    frames, pools = [], {}
    for visitor, history in context.histories.items():
        candidates = generate_candidates(visitor, history, collaborative, content)
        pools[visitor] = candidates
        features = build_rank_features(visitor, candidates, snapshot.as_of_time, context)
        features["role"] = snapshot.role
        features["snapshot"] = snapshot.as_of_time.isoformat()
        features["outcome_end"] = snapshot.outcome_end.isoformat()
        features["visitor_id"] = visitor
        features["label"] = [int((visitor, item) in positive_pairs) for item in features.item_id]
        features["exposed_in_window"] = [int((visitor, item) in exposed_pairs) for item in features.item_id]
        features["negative_provenance"] = np.where(features.label.eq(1), "positive",
            np.where(features.exposed_in_window.eq(1), "exposed", "unobserved_proxy"))
        frames.append(features.loc[:, ROW_COLUMNS])
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ROW_COLUMNS)

    def pool_for(visitor, history):
        if visitor in pools:
            return pools[visitor]
        return generate_candidates(visitor, history, collaborative, content)

    recalls, missed_pairs = {}, []
    for source in ("union", *SOURCE_COLUMNS):
        def retrieve(visitor, history, limit):
            pool = pool_for(visitor, history)
            return pool if source == "union" else pool.loc[pool[SOURCE_COLUMNS[source][1]].eq(1)]

        evaluation = evaluate_snapshot(snapshot, events, items, retrieve)
        if source == "union":
            missed_pairs = [{"visitor_id": row["visitor_id"], "item_id": item,
                             "segment": row["segment"]}
                            for row in evaluation["visitor_results"]
                            for item in sorted(set(row["relevant_items"]) - set(row["candidate_items"]))]
        segments = {}
        for segment, summary in evaluation["segments"].items():
            measured = [row for row in evaluation["visitor_results"] if row["segment"] == segment]
            hits = sum(len(set(row["candidate_items"]) & set(row["relevant_items"])) for row in measured)
            total = sum(len(row["relevant_items"]) for row in measured)
            segments[segment] = {
                "visitors": summary["visitors"], "evaluated_visitors": summary["evaluated_visitors"],
                "macro_recall_at_50": summary["candidate_recall_at_50"],
                "retrieved_positive_pairs": hits, "relevant_positive_pairs": total,
                "micro_recall": hits / total if total else None,
            }
        recalls[source] = segments
    flags = [pair[1] for pair in SOURCE_COLUMNS.values()]
    flag_count = rows[flags].sum(axis=1)
    contributions = {}
    for source, (_, flag) in SOURCE_COLUMNS.items():
        nominated = rows[flag].eq(1)
        exclusive = nominated & flag_count.eq(1)
        contributions[source] = {
            "nominations": int(nominated.sum()), "exclusive_candidates": int(exclusive.sum()),
            "positive_nominations": int((nominated & rows.label.eq(1)).sum()),
            "exclusive_positive_pairs": int((exclusive & rows.label.eq(1)).sum()),
        }
    overlaps = {}
    sources = list(SOURCE_COLUMNS)
    for i, first in enumerate(sources):
        for second in sources[i + 1:]:
            overlaps[f"{first}+{second}"] = int((rows[SOURCE_COLUMNS[first][1]].eq(1)
                & rows[SOURCE_COLUMNS[second][1]].eq(1)).sum())
    sizes = {visitor: len(pool) for visitor, pool in pools.items()}
    diagnostics = {
        "role": snapshot.role, "snapshot": snapshot.as_of_time.isoformat(),
        "outcome_end": snapshot.outcome_end.isoformat(), "rows": len(rows),
        "positive_rows": int(rows.label.eq(1).sum()),
        "negative_rows": int(rows.label.eq(0).sum()),
        "exposed_negative_rows": int((rows.label.eq(0) & rows.exposed_in_window.eq(1)).sum()),
        "pool_sizes": {"visitors": len(sizes), "min": min(sizes.values(), default=0),
                       "max": max(sizes.values(), default=0),
                       "mean": sum(sizes.values()) / len(sizes) if sizes else 0,
                       "per_visitor": sizes},
        "recall": recalls, "missed_positive_pairs": missed_pairs,
        "contributions": contributions, "pairwise_overlap": overlaps,
        "example": None,
    }
    if len(rows):
        positives = rows.loc[rows.label.eq(1)]
        example = (positives if len(positives) else rows).iloc[0].to_dict()
        diagnostics["example"] = _trace(example, context, outcomes, exposures)
    return rows, diagnostics


def sample_training_rows(rows, negative_ratio=3, seed=42):
    """Keep all positives; sample exposed negatives first at a snapshot-wide ratio.

    Sorted inputs make selection independent of input ordering. Each snapshot
    uses the seed independently. Zero-positive snapshots yield no sampled rows.
    """
    if not isinstance(negative_ratio, int) or isinstance(negative_ratio, bool) or negative_ratio < 1:
        raise ValueError("negative_ratio must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not set(ROW_COLUMNS) <= set(rows):
        raise ValueError("Rows must have the rank dataset schema")
    if not rows.role.eq("train").all():
        raise ValueError("Only training rows may be sampled")
    if not rows.label.isin([0, 1]).all() or not rows.exposed_in_window.isin([0, 1]).all():
        raise ValueError("Labels and exposure flags must be binary")
    if rows.duplicated(["snapshot", "visitor_id", "item_id"]).any():
        raise ValueError("Duplicate snapshot/visitor/item rows")
    selected, diagnostics = [], []
    for cutoff, frame in rows.groupby("snapshot", sort=True):
        frame = frame.sort_values(["visitor_id", "item_id"]).reset_index(drop=True)
        positives = frame.loc[frame.label.eq(1)]
        target = len(positives) * negative_ratio
        rng = np.random.default_rng(seed)
        parts, remaining = [positives], target
        for exposed in (1, 0):
            available = frame.loc[frame.label.eq(0) & frame.exposed_in_window.eq(exposed)]
            count = min(remaining, len(available))
            indices = rng.choice(len(available), size=count, replace=False)
            parts.append(available.iloc[indices])
            remaining -= count
        sample = pd.concat(parts).sort_values(["visitor_id", "item_id"])
        selected.append(sample)
        negatives = sample.loc[sample.label.eq(0)]
        diagnostics.append({
            "snapshot": cutoff, "positive_rows": len(positives), "negative_rows": len(negatives),
            "requested_negative_rows": target, "negative_shortfall": remaining,
            "achieved_ratio": len(negatives) / len(positives) if len(positives) else None,
            "exposed_negative_rows": int(negatives.exposed_in_window.eq(1).sum()),
            "proxy_negative_rows": int(negatives.exposed_in_window.eq(0).sum()),
            "status": "no_retrieved_positives" if positives.empty else "sampled",
        })
    result = pd.concat(selected, ignore_index=True) if selected else rows.iloc[:0].copy()
    return result, diagnostics


def render_candidate_report(metadata, command):
    """Report retrieval/sampling only; item-ID order is not a ranking baseline."""
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    lines = ["# Phase 5 candidate analysis", "", "## Reproduce", "", "```text", command, "```", "",
        "History is strictly before each UTC cutoff; labels and exposure provenance use the following "
        "seven days, including the cutoff and excluding the outcome end. Models are refitted at every "
        "snapshot. Visitors in exported datasets have pre-cutoff events; visitors without future "
        "positives remain eligible for negative sampling. Validation is unsampled; final test is reserved.", "",
        "The pool contains up to 20 collaborative, 20 content, and 10 popularity nominations, "
        "deduplicated by item ID. Rows are ordered by item ID, not recommendation quality. "
        "Missing source scores are 0.0 with flag 0; nominated zero scores have flag 1. "
        "Cold start uses only ten popularity nominations.", "", "## Union retrieval", "",
        "Recall uses recommendable future carts/purchases, excluding unavailable items and prior "
        "purchases. Macro recall averages eligible visitors with positives; micro recall divides "
        "retrieved positive pairs by all relevant pairs. Empty cohorts are unavailable.", "",
        "| Role / cutoff | Visitors / evaluated | Pool min / mean / max | Positive pairs | Macro Recall@50 | Micro recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for d in metadata["snapshots"]:
        r, p = d["recall"]["union"]["personalized"], d["pool_sizes"]
        lines.append(f"| {d['role']} / {d['snapshot'][:10]} | {p['visitors']} / {r['evaluated_visitors']} "
            f"| {p['min']} / {p['mean']:.2f} / {p['max']} "
            f"| {r['retrieved_positive_pairs']} / {r['relevant_positive_pairs']} "
            f"| {number(r['macro_recall_at_50'])} | {number(r['micro_recall'])} |")
    lines += ["", "**Retrieval ceiling:** the ranker cannot rescue positives omitted from this pool. "
        "Phases 3/4 used 50 candidates per individual source; their earlier 100% validation recall "
        "is not a like-for-like comparison with this smaller budget. Inspect these misses before "
        "Phase 6; this phase does not tune generators or increase budgets.", "",
        "### Missed recommendable positives", ""]
    for d in metadata["snapshots"]:
        misses = ", ".join(f"`{pair['visitor_id']} / {pair['item_id']}` ({pair['segment']})"
                           for pair in d["missed_positive_pairs"])
        lines.append(f"- {d['snapshot'][:10]}: {misses or 'none'}.")
    lines += ["", "## Source contributions and overlap", "",
        "All nomination/overlap counts below cover every historical visitor. Exclusive candidates "
        "have exactly one source; pairwise intersections include three-source candidates.", "",
        "| Cutoff | Source | Macro Recall@50 | Nominations | Exclusive candidates | Positive nominations | Exclusive positives |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for d in metadata["snapshots"]:
        for source, c in d["contributions"].items():
            recall = d["recall"][source]["personalized"]["macro_recall_at_50"]
            lines.append(f"| {d['snapshot'][:10]} | {source} | {number(recall)} | {c['nominations']} "
                f"| {c['exclusive_candidates']} | {c['positive_nominations']} | {c['exclusive_positive_pairs']} |")
    lines += ["", "| Cutoff | Collab + content | Collab + popularity | Content + popularity |", "| --- | ---: | ---: | ---: |"]
    for d in metadata["snapshots"]:
        o = d["pairwise_overlap"]
        lines.append(f"| {d['snapshot'][:10]} | {o['collaborative+content']} | {o['collaborative+popularity']} | {o['content+popularity']} |")
    lines += ["", "### Cold start", ""]
    for d in metadata["snapshots"]:
        r = d["recall"]["union"]["cold_start"]
        lines.append(f"- {d['snapshot'][:10]}: {r['visitors']} observed visitors, {r['evaluated_visitors']} "
                     f"with recommendable positives; macro recall {number(r['macro_recall_at_50'])}.")
    lines += ["", "## Training sampling", "",
        f"Seed: **{metadata['seed']}**. Requested negatives per positive: **{metadata['negative_ratio']}** "
        "per training snapshot. Keep all retrieved positives; sample exposed negatives first, then "
        "unobserved proxies. Exposure means a visitor/item impression in the outcome window; a "
        "view-only outcome still has label 0. A negative indicates no observed target action, not dislike. "
        "Sampled prevalence is not representative exposure prevalence; future model scores are "
        "ordering signals rather than calibrated probabilities.", "",
        "| Cutoff | Available negatives / exposed | Kept positives | Sampled negatives / exposed / proxy | Achieved ratio | Shortfall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    by_snapshot = {d["snapshot"]: d for d in metadata["snapshots"]}
    for s in metadata["sampling"]:
        d = by_snapshot[s["snapshot"]]
        lines.append(f"| {s['snapshot'][:10]} | {d['negative_rows']} / {d['exposed_negative_rows']} "
            f"| {s['positive_rows']} | {s['negative_rows']} / {s['exposed_negative_rows']} / {s['proxy_negative_rows']} "
            f"| {number(s['achieved_ratio'])} | {s['negative_shortfall']} |")
        if s["status"] == "no_retrieved_positives":
            lines.append(f"\nSnapshot {s['snapshot']} contributes no sampled rows because no positives were retrieved.\n")
    # Empty candidate pools cannot appear in a groupby-based sampling result.
    for d in metadata["snapshots"]:
        if d["role"] == "train" and d["rows"] == 0:
            lines.append(f"\nSnapshot {d['snapshot']} has an empty candidate pool and contributes no training rows.\n")
    lines += ["", f"Exported rows: **{metadata['training_rows']} training**, **{metadata['validation_rows']} validation**.",
              "", "## Feature and label trace", ""]
    example = next((d["example"] for d in metadata["snapshots"] if d["example"]), None)
    if example:
        row = example["row"]
        lines += [f"Visitor `{row['visitor_id']}`, item `{row['item_id']}`, snapshot **{row['snapshot']}**.", "",
                  "| Feature | Value | Information time |", "| --- | ---: | --- |"]
        for name in FEATURE_COLUMNS:
            value = row[name]
            time = "Catalog available at cutoff (static metadata)" if name == "item_price" else "Events/models strictly before cutoff"
            if name == "visitor_item_price_gap":
                time += "; static available prices"
            lines.append(f"| {name} | {'missing' if value is None else value} | {time} |")
        lines += ["", f"Last visitor activity: **{example['history_last_timestamp']}**. Category "
            f"`{example['candidate_category']}` strength is {example['category_weighted_strength']} / "
            f"{example['visitor_weighted_strength']}; weighted mean historical price is "
            f"{example['visitor_weighted_mean_price']:.6f}. Item creation: {example['item_created_at']}.", "",
            f"Label: **{row['label']}**, from target actions in **[{row['snapshot']}, {row['outcome_end']})**. "
            f"Target timestamps: {', '.join(example['target_outcome_timestamps']) or 'none'}. "
            f"Exposure provenance: `{row['negative_provenance']}`; impression timestamps: "
            f"{', '.join(example['impression_timestamps']) or 'none'}.", ""]
    lines += ["IDs, role, snapshot, outcome end, label, exposure flag, and provenance are metadata, "
        "excluded from FEATURE_COLUMNS. Recency and price gap are missing for empty histories; "
        "Phase 6 will impute and scale features. Catalog attributes are assumed static because this "
        "dataset has no price/category revision history. Synthetic exposure is incomplete evidence "
        "and does not establish real-world preference or business uplift.", "",
        "**Checkpoint:** source flags describe nomination, not comparable score scales. Features "
        "describe the past; labels describe the following seven days. Improving ranking only "
        "reorders retrieved items, so missing relevant candidates require retrieval changes.", ""]
    return "\n".join(lines)


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build train/validation CSVs and candidate analysis")
    build.add_argument("--db", type=Path, default=DEFAULT_DB)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--negative-ratio", type=int, default=3)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        # Validate arguments before expensive fits or any output writes.
        sample_training_rows(pd.DataFrame(columns=ROW_COLUMNS), args.negative_ratio, args.seed)
        paths = [args.output_dir / name for name in ("train.csv", "validation.csv", "metadata.json")]
        paths.append(args.report)
        resolved = [path.resolve() for path in paths]
        if args.db.resolve() in resolved or len(set(resolved)) != len(resolved):
            raise ValueError("Output paths must be distinct and must not overwrite the database")
        events, items, impressions = load_events(args.db), load_items(args.db), load_impressions(args.db)
        end = as_utc(DEFAULT_OBSERVATION_END)
        if any((frame.timestamp >= end).any() for frame in (events, impressions)):
            raise ValueError("Observed activity extends beyond the declared observation end")
        snapshots = [s for s in build_snapshots() if s.role != "test"]
        frames, diagnostics = [], []
        for snapshot in snapshots:
            rows, diagnostic = build_snapshot_dataset(snapshot, events, items, impressions)
            frames.append(rows)
            diagnostics.append(diagnostic)
        all_rows = pd.concat(frames, ignore_index=True)
        training, sampling = sample_training_rows(all_rows.loc[all_rows.role.eq("train")],
                                                   args.negative_ratio, args.seed)
        sampled_snapshots = {s["snapshot"] for s in sampling}
        for d in diagnostics:
            if d["role"] == "train" and d["snapshot"] not in sampled_snapshots:
                sampling.append({"snapshot": d["snapshot"], "positive_rows": 0, "negative_rows": 0,
                    "requested_negative_rows": 0, "negative_shortfall": 0, "achieved_ratio": None,
                    "exposed_negative_rows": 0, "proxy_negative_rows": 0,
                    "status": "no_retrieved_positives"})
        sampling.sort(key=lambda s: s["snapshot"])
        validation = all_rows.loc[all_rows.role.eq("validation")].reset_index(drop=True)
        metadata = {
            "schema_version": 1, "database": str(args.db), "seed": args.seed,
            "negative_ratio": args.negative_ratio, "sampling_unit": "snapshot",
            "candidate_budgets": CANDIDATE_BUDGETS, "event_weights": EVENT_WEIGHTS,
            "missing_source_score": 0.0, "feature_columns": list(FEATURE_COLUMNS),
            "training_rows": len(training), "validation_rows": len(validation),
            "snapshots": diagnostics, "sampling": sampling, "test_reserved": True,
        }
        def quote(value):
            return "'" + str(value).replace("'", "''") + "'"

        command = (f"uv run python -B -m src.rank_dataset build --db {quote(args.db)} "
                   f"--seed {args.seed} --negative-ratio {args.negative_ratio} "
                   f"--output-dir {quote(args.output_dir)} --report {quote(args.report)}")
        report = render_candidate_report(metadata, command)
        # Metadata is written last; each individual file is replaced atomically.
        for path, content in ((paths[0], training.to_csv(index=False)),
                              (paths[1], validation.to_csv(index=False)), (paths[3], report),
                              (paths[2], json.dumps(metadata, indent=2, allow_nan=False) + "\n")):
            _atomic_write(path, content)
    except (ValueError, TypeError, OSError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        parser.exit(1, f"Rank dataset build failed: {exc}\n")
    print(json.dumps({"training_rows": len(training), "validation_rows": len(validation),
                      "output_dir": str(args.output_dir), "report": str(args.report),
                      "sampling": sampling, "test_reserved": True}, indent=2))


if __name__ == "__main__":
    main()
