"""Rebuild older snapshot rows, tune logistic ranking, and compare on validation."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import sqlite3
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning

from src.candidates import CANDIDATE_BUDGETS, generate_candidates
from src.collaborative import fit_collaborative
from src.content_based import fit_content_based
from src.data import DEFAULT_DB, EVENT_WEIGHTS, as_utc, load_events, load_impressions, load_items
from src.evaluate import evaluate_snapshot
from src.rank_dataset import ROW_COLUMNS, _atomic_write, build_snapshot_dataset, sample_training_rows
from src.rank_features import FEATURE_COLUMNS, build_rank_features, prepare_rank_context
from src.ranker import (BLEND_WEIGHTS, METHODS, ModelBundle, blend_scores, fit_ranker, full_source_scores,
                        rank_pool, save_bundle)
from src.split import DEFAULT_OBSERVATION_END, build_snapshots


DEFAULT_ARTIFACT = Path("artifacts/model_bundle.pkl")
DEFAULT_REPORT = Path("reports/model_comparison.md")


def selection_key(trial):
    metrics = trial["evaluation"]["segments"]["personalized"]
    return (-metrics["map_at_10"], -metrics["recall_at_10"], trial["C"],
            trial["class_weight"] is not None)


def prepare_inputs(context, collaborative, content, visitor_id, history):
    candidates = generate_candidates(visitor_id, history, collaborative, content)
    features = build_rank_features(visitor_id, candidates, context.as_of_time, context)
    sources = full_source_scores(candidates, visitor_id, history, collaborative, content)
    return candidates, features, sources


def run_training(events, items, impressions=None, seed=42, negative_ratio=3,
                 model_choice="logistic", timing_repetitions=20):
    """No outputs are written until data building, selection, and scoring succeed."""
    sample_training_rows(pd.DataFrame(columns=ROW_COLUMNS), negative_ratio, seed)
    if seed >= 2**32:
        raise ValueError("seed must be less than 2**32")
    if model_choice not in ("logistic", "blend"):
        raise ValueError("model choice must be logistic or blend")
    if not isinstance(timing_repetitions, int) or timing_repetitions < 1:
        raise ValueError("timing repetitions must be positive")
    end = as_utc(DEFAULT_OBSERVATION_END)
    for frame in (events, impressions):
        if frame is not None and (frame.timestamp >= end).any():
            raise ValueError("Observed activity extends beyond the declared observation end")
    snapshots = [s for s in build_snapshots() if s.role != "test"]
    frames, diagnostics = [], []
    for snapshot in snapshots:
        rows, diagnostic = build_snapshot_dataset(snapshot, events, items, impressions)
        frames.append(rows)
        diagnostics.append(diagnostic)
    all_rows = pd.concat(frames, ignore_index=True)
    training, sampling = sample_training_rows(all_rows.loc[all_rows.role.eq("train")],
                                              negative_ratio, seed)
    validation = all_rows.loc[all_rows.role.eq("validation")].reset_index(drop=True)
    snapshot = snapshots[-1]
    training_ends = pd.to_datetime(training.outcome_end, utc=True)
    if (training_ends > snapshot.as_of_time).any():
        raise ValueError("Training labels overlap validation")
    context = prepare_rank_context(events, items, snapshot.as_of_time)
    collaborative = fit_collaborative(context.history, items, snapshot.as_of_time)
    content = fit_content_based(context.history, items, snapshot.as_of_time)
    cache = {visitor: prepare_inputs(context, collaborative, content, visitor, history)
             for visitor, history in context.histories.items()}
    # Verify the inference path reproduces the freshly rebuilt Phase 5 features.
    for visitor, (_, features, _) in cache.items():
        expected = validation.loc[validation.visitor_id.eq(visitor), ["item_id", *FEATURE_COLUMNS]]
        pd.testing.assert_frame_equal(features.reset_index(drop=True), expected.reset_index(drop=True))

    def callback(method, ranker=None):
        def recommend(visitor, history, limit):
            if visitor not in cache:
                cache[visitor] = prepare_inputs(context, collaborative, content, visitor, history)
            pool, features, sources = cache[visitor]
            selected = "popularity" if history.empty else method
            return rank_pool(selected, pool, features, sources, ranker)
        return recommend

    trials, successful = [], []
    available = not training.empty and training.label.nunique() == 2
    for C in (0.01, 0.1, 1.0, 10.0) if available else ():
        for weight in (None, "balanced"):
            trial = {"C": C, "class_weight": weight}
            try:
                ranker = fit_ranker(training, C, weight, seed)
            except ConvergenceWarning as exc:
                trial.update(status="nonconverged", error=str(exc))
            else:
                evaluation = evaluate_snapshot(snapshot, events, items, callback("logistic", ranker))
                if evaluation["segments"]["personalized"]["map_at_10"] is None:
                    trial.update(status="no_validation_cohort")
                else:
                    trial.update(status="ok", evaluation=evaluation)
                    successful.append((trial, ranker))
            trials.append(trial)
    winner = min(successful, key=lambda pair: selection_key(pair[0])) if successful else None
    ranker = winner[1] if winner else None
    unavailable = None if winner else ("Training requires both classes and nonempty sampled rows"
                                      if not available else "No successful validation configuration")
    if ranker is None and model_choice == "logistic":
        raise ValueError(f"{unavailable}; select --model blend explicitly")
    comparisons, timings = {}, {}
    for method in METHODS:
        if method == "logistic" and ranker is None:
            continue
        comparisons[method] = evaluate_snapshot(snapshot, events, items, callback(method, ranker))
        measured = []
        for visitor in sorted(context.histories):
            pool, features, sources = cache[visitor]
            def score():
                if method == "logistic":
                    return ranker.score(features)
                if method == "blend":
                    return blend_scores(sources)
                column = {"popularity": "popularity_score", "collaborative": "collab_score",
                          "content": "content_score"}[method]
                return sources[column].to_numpy(dtype=float)
            score()
            for _ in range(timing_repetitions):
                started = perf_counter()
                score()
                measured.append((perf_counter() - started) * 1000)
        timings[method] = {"median_ms": float(np.median(measured)) if measured else None,
                           "p95_ms": float(np.percentile(measured, 95)) if measured else None,
                           "samples": len(measured), "repetitions": timing_repetitions}
    metadata = {
        "serving_cutoff": snapshot.as_of_time.isoformat(), "item_ids": list(collaborative.item_ids),
        "training_snapshots": [{"snapshot": s.as_of_time.isoformat(), "outcome_end": s.outcome_end.isoformat()}
                               for s in snapshots if s.role == "train"],
        "validation_outcome_end": snapshot.outcome_end.isoformat(), "seed": seed,
        "negative_ratio": negative_ratio, "candidate_budgets": dict(CANDIDATE_BUDGETS),
        "event_weights": dict(EVENT_WEIGHTS), "blend_weights": dict(BLEND_WEIGHTS),
        "feature_columns": list(FEATURE_COLUMNS), "training_rows": len(training),
        "training_positives": int(training.label.sum()), "validation_rows": len(validation),
        "validation_positives": int(validation.label.sum()), "sampling": sampling,
        "selected_hyperparameters": {"C": winner[0]["C"], "class_weight": winner[0]["class_weight"]} if winner else None,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "test_reserved": True,
    }
    serving_collaborative = replace(collaborative, matrix=None, visitor_ids=())
    bundle = ModelBundle(model_choice, ranker, serving_collaborative, content, metadata).validate()
    data = {"metadata": metadata, "trials": trials, "comparisons": comparisons, "timings": timings,
            "logistic_unavailable": unavailable, "diagnostics": diagnostics, "selected_model": model_choice}
    data["examples"] = []
    evaluated = comparisons["popularity"]["visitor_results"]
    known_examples = [r for r in evaluated if r["segment"] == "personalized"]
    changed = [r for r in known_examples if callback(model_choice, ranker)(r["visitor_id"],
               context.histories[r["visitor_id"]], 50).head(10).item_id.tolist() != r["recommendations"]]
    example = next(iter(changed or known_examples), None)
    if example:
        visitor = example["visitor_id"]
        pool, features, sources = cache[visitor]
        ranked = callback(model_choice, ranker)(visitor, context.histories[visitor], 50)
        if len(ranked):
            item = ranked.iloc[0].item_id
            row = features.loc[features.item_id.eq(item)].iloc[0]
            trace = {name: None if pd.isna(row[name]) else float(row[name]) for name in FEATURE_COLUMNS}
            data["examples"].append({"visitor_id": visitor, "item_id": item, "features": trace,
                "score": float(ranked.iloc[0].score), "relevant_items": example["relevant_items"],
                "rankings": {method: callback(method, ranker)(visitor, context.histories[visitor], 50)
                             .head(10).item_id.tolist() for method in comparisons}})
    return bundle, data


def render_report(bundle, data, command):
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"
    metadata = bundle.metadata
    lines = ["# Phase 6 model comparison", "", "## Reproduce", "", "```text", command, "```", "",
             f"Explicit configured model: **{bundle.selected_model}**. Seed: {metadata['seed']}; "
             f"negatives per positive: {metadata['negative_ratio']}.", "",
             f"Training: **{metadata['training_rows']} rows / {metadata['training_positives']} positives**. "
             f"Validation: **{metadata['validation_rows']} unsampled rows / {metadata['validation_positives']} positives**.", "",
             "Training cutoffs: " + ", ".join(s["snapshot"][:10] for s in metadata["training_snapshots"]) + ".",
             f"Validation history is strictly before **{metadata['serving_cutoff']}**; target outcomes are in "
             f"[{metadata['serving_cutoff']}, {metadata['validation_outcome_end']}). Final test remains reserved.", "",
             "## Validation tuning", "", "Imputation, scaling, and coefficients use training rows only. "
             "Selection uses MAP@10, then Recall@10, smaller C, then no class weighting. "
             "The saved pipeline is not refitted on validation labels.", "",
             "| C | Class weight | Status | MAP@10 | Recall@10 |", "| ---: | --- | --- | ---: | ---: |"]
    for trial in data["trials"]:
        metrics = trial.get("evaluation", {}).get("segments", {}).get("personalized", {})
        lines.append(f"| {trial['C']} | {trial['class_weight'] or 'none'} | {trial['status']} | "
                     f"{number(metrics.get('map_at_10'))} | {number(metrics.get('recall_at_10'))} |")
    lines += ["", f"Selected logistic configuration: `{metadata['selected_hyperparameters']}`."]
    if data["logistic_unavailable"]:
        lines += ["", "Logistic unavailable: " + data["logistic_unavailable"] + ". Blend was explicitly selected."]
    lines += ["", "## Same-pool comparisons", "",
              "All methods reorder the same deduplicated 20 collaborative / 20 content / 10 popularity "
              "union. Purchased and unavailable items are excluded; no category cap is applied. "
              "Individual-source baselines use actual scores for every union item, not missing-nomination "
              "zeros. The logistic inputs retain Phase 5 nomination scores and source flags.", "",
              "Blend: 0.4 collaborative + 0.4 content + 0.2 popularity, each source min–max normalized "
              "within the visitor's fixed pool. Constant sources contribute zero. Ties use item ID.", ""]
    for segment in ("personalized", "cold_start"):
        lines += [f"### {segment.replace('_', ' ').title()}", "",
                  "| Method | Visitors / evaluated | Candidate Recall@50 | Precision@10 | Recall@10 | MAP@10 | Coverage |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method, evaluation in data["comparisons"].items():
            m = evaluation["segments"][segment]
            lines.append(f"| {method} | {m['visitors']} / {m['evaluated_visitors']} | " + " | ".join(
                number(m[name]) for name in ("candidate_recall_at_50", "precision_at_10", "recall_at_10",
                                            "map_at_10", "catalog_coverage")) + " |")
        lines += [""]
    lines += ["Cold start always uses popularity because training rows have prior history. "
              "Only visitors with recommendable held-out carts/purchases enter quality averages; "
              "unavailable cohorts are not scored as zero.", "", "### Scoring latency", "",
              f"Prepared feature/source-score rows; one warm-up and {next(iter(data['timings'].values()))['repetitions']} "
              "repetitions per historical visitor. "
              "Times include scoring only, excluding sorting, retrieval, features, fitting, and metrics. "
              "Source baselines read prepared scores; this is not whole-request or HTTP latency.", "",
              "| Method | Samples | Median (ms) | p95 (ms) |", "| --- | ---: | ---: | ---: |"]
    for method, timing in data["timings"].items():
        lines.append(f"| {method} | {timing['samples']} | {number(timing['median_ms'])} | {number(timing['p95_ms'])} |")
    if bundle.ranker is not None:
        ml = data["comparisons"]["logistic"]["segments"]["personalized"]
        lines += ["", "### Logistic differences from simpler methods", "",
                  "| Compared with | Δ Precision@10 | Δ Recall@10 | Δ MAP@10 | Δ Coverage |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for method in METHODS[:-1]:
            baseline = data["comparisons"][method]["segments"]["personalized"]
            lines.append(f"| {method} | " + " | ".join(number(ml[name] - baseline[name]) for name in
                ("precision_at_10", "recall_at_10", "map_at_10", "catalog_coverage")) + " |")
        coefficients = bundle.ranker.pipeline.named_steps["model"].coef_[0]
        lines += ["", "## Coefficients and learning checkpoint", "",
                  "Coefficients change log-odds per one training standard deviation of an imputed feature, "
                  "holding other inputs fixed. Positive coefficients raise the score and negative coefficients "
                  "lower it; correlated features and regularization prevent causal interpretations.", "",
                  "| Feature | Standardized coefficient | Interpretation |", "| --- | ---: | --- |"]
        descriptions = {"collab_score": "Stronger collaborative nomination", "content_score": "Stronger metadata similarity nomination",
                        "visitor_category_affinity": "Greater historical affinity to the item's category"}
        for name, value in zip(FEATURE_COLUMNS, coefficients):
            direction = "raises" if value > 0 else "lowers" if value < 0 else "does not change"
            meaning = descriptions.get(name, "Increasing this feature")
            lines.append(f"| {name} | {value:.6f} | {meaning} {direction} the conditional ordering score. |")
        baselines = {method: data["comparisons"][method]["segments"]["personalized"]["map_at_10"] for method in METHODS[:-1]}
        best = max(baselines, key=baselines.get)
        outcome = "exceeds" if ml["map_at_10"] > baselines[best] else "matches" if ml["map_at_10"] == baselines[best] else "is below"
        lines += ["", f"Measured result: logistic MAP@10 ({ml['map_at_10']:.6f}) {outcome} the best simpler "
                  f"method, {best} ({baselines[best]:.6f}). Model choice remains explicit."]
    for example in data["examples"]:
        lines += ["", "## Trace one ranking", "", f"Visitor `{example['visitor_id']}`; relevant items: "
                  + ", ".join(example["relevant_items"]) + ".", "", "| Method | Top 10 in order |", "| --- | --- |"]
        for method, ids in example["rankings"].items():
            lines.append(f"| {method} | {', '.join(ids)} |")
        lines += ["", f"Configured model's first item: `{example['item_id']}`, score **{example['score']:.6f}**.", "",
                  "| Feature | Pre-cutoff value |", "| --- | ---: |"]
        lines += [f"| {name} | {number(value)} |" for name, value in example["features"].items()]
        if bundle.ranker is not None and bundle.selected_model == "logistic":
            frame = pd.DataFrame([example["features"]])
            transformed = bundle.ranker.pipeline[:-1].transform(frame)
            model = bundle.ranker.pipeline.named_steps["model"]
            logit = float(model.intercept_[0] + transformed[0] @ model.coef_[0])
            lines += ["", f"After training-fitted imputation/scaling, intercept + coefficient contributions "
                      f"= **{logit:.6f}**; applying the logistic sigmoid gives the score above."]
    lines += ["", "## Assumptions and limitations", "",
              "Label 1 means an observed cart or purchase in the following seven days; absence of an action "
              f"is an offline proxy, not proof of dislike. Exposure-first {metadata['negative_ratio']}:1 sampling changes prevalence; "
              "class weighting is compared on unsampled validation, with no inverse-sampling correction. "
              "Scores are not calibrated purchase probabilities. Logistic regression fits this binary target "
              "with regularization; ridge regression's squared-error numeric objective is a weaker first choice.", "",
              "IDs, labels, snapshot times, and future exposure provenance are excluded from model inputs. "
              "Feature/generator events are strictly before each cutoff; catalog metadata is static. "
              "The ranker cannot recover the missed relevant items outside its candidate pool. "
              "The small synthetic validation cohort and eight tuning choices limit confidence; "
              "offline improvements do not prove business uplift.", "",
              "The bundle contains the full preprocessing pipeline and schema, bounded collaborative serving "
              "state, fitted content representation, and training provenance. Its generators use the validation "
              "cutoff; it is not a later serving refit. Raw visitor histories are read separately. "
              "Only trusted local pickle bundles should be loaded.", "",
              "Versions: `" + json.dumps(metadata["versions"], sort_keys=True) + "`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", choices=("logistic", "blend"), default="logistic")
    args = parser.parse_args()
    try:
        sample_training_rows(pd.DataFrame(columns=ROW_COLUMNS), args.negative_ratio, args.seed)
        paths = [args.db.resolve(), args.artifact.resolve(), args.report.resolve()]
        if len(set(paths)) != len(paths):
            raise ValueError("Database, artifact, and report paths must be distinct")
        events, items, impressions = load_events(args.db), load_items(args.db), load_impressions(args.db)
        bundle, data = run_training(events, items, impressions, args.seed, args.negative_ratio, args.model)
        bundle.metadata["database"] = str(args.db)
        def quote(value):
            return "'" + str(value).replace("'", "''") + "'"
        command = (f"uv run python -B -m src.train --db {quote(args.db)} --seed {args.seed} "
                   f"--negative-ratio {args.negative_ratio} --model {args.model} "
                   f"--artifact {quote(args.artifact)} --report {quote(args.report)}")
        report = render_report(bundle, data, command)
        save_bundle(bundle, args.artifact)
        _atomic_write(args.report, report)
    except (ValueError, TypeError, OSError, sqlite3.Error, pd.errors.DatabaseError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    print(json.dumps({"artifact": str(args.artifact), "report": str(args.report),
                      "selected_model": bundle.selected_model, "metadata": bundle.metadata,
                      "validation": {method: result["segments"] for method, result in data["comparisons"].items()},
                      "test_reserved": True}, indent=2))


if __name__ == "__main__":
    main()
