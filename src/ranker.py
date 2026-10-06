"""Cutoff-safe logistic ranking, fixed baselines, and local serving bundles."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import pickle
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.candidates import CANDIDATE_BUDGETS, generate_candidates
from src.data import EVENT_WEIGHTS, as_utc
from src.rank_features import FEATURE_COLUMNS, build_rank_features


METHODS = ("popularity", "collaborative", "content", "blend", "logistic")
BLEND_WEIGHTS = {"collab_score": 0.4, "content_score": 0.4, "popularity_score": 0.2}
BUNDLE_VERSION = 1


def _feature_matrix(rows):
    if not isinstance(rows, pd.DataFrame):
        raise ValueError("feature rows must be a DataFrame")
    for name in FEATURE_COLUMNS:
        if list(rows.columns).count(name) != 1:
            raise ValueError(f"Required feature must occur exactly once: {name}")
        if not pd.api.types.is_numeric_dtype(rows[name]):
            raise ValueError(f"Feature must be numeric: {name}")
    matrix = rows.loc[:, list(FEATURE_COLUMNS)].astype(float)
    if np.isinf(matrix.to_numpy()).any():
        raise ValueError("Features must not contain infinity")
    return matrix


@dataclass
class Ranker:
    pipeline: Pipeline = field(repr=False)
    feature_columns: tuple = FEATURE_COLUMNS

    def score(self, feature_rows):
        if tuple(self.feature_columns) != FEATURE_COLUMNS:
            raise ValueError("Unsupported ranker feature schema")
        matrix = _feature_matrix(feature_rows)
        if matrix.empty:
            return np.empty(0, dtype=float)
        scores = self.pipeline.predict_proba(matrix)[:, 1]
        if not np.isfinite(scores).all():
            raise ValueError("Ranker produced nonfinite scores")
        return scores


def fit_ranker(training_rows, C=1.0, class_weight=None, seed=42):
    """Fit only older snapshot rows; labels and provenance cannot enter X."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not np.isfinite(C) or C <= 0 or class_weight not in (None, "balanced"):
        raise ValueError("Invalid regularization strength or class weighting")
    matrix = _feature_matrix(training_rows)
    if "label" not in training_rows or list(training_rows.columns).count("label") != 1:
        raise ValueError("Training rows require one binary label column")
    labels = training_rows.label
    if matrix.empty or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("Logistic training requires nonempty rows with both binary classes")
    if "role" in training_rows and not training_rows.role.eq("train").all():
        raise ValueError("Only training snapshots may fit the ranker")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, class_weight=class_weight, solver="lbfgs",
                                     max_iter=2000, random_state=seed)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(matrix, labels.astype(int))
    return Ranker(pipeline)


def full_source_scores(candidates, visitor_id, history, collaborative, content):
    """Score the fixed union, without enlarging it or using nomination zeros."""
    result = pd.DataFrame({"item_id": candidates.item_id}).reset_index(drop=True)
    for model, column in ((collaborative, "collab_score"), (content, "content_score")):
        generated = model.generate(visitor_id, history, max(1, len(model.item_ids)))
        scores = generated.set_index("item_id").score
        result[column] = result.item_id.map(scores).fillna(0.0).astype(float)
    result["popularity_score"] = result.item_id.map(collaborative.popularity_scores).fillna(0.0)
    return result


def blend_scores(source_scores):
    result = np.zeros(len(source_scores), dtype=float)
    for column, weight in BLEND_WEIGHTS.items():
        values = source_scores[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Source scores must be finite")
        if len(values) and values.max() > values.min():
            result += weight * (values - values.min()) / (values.max() - values.min())
    return result


def rank_pool(method, candidates, features, source_scores, ranker=None):
    """Return the whole ordered pool; retrieval recall is independent of method."""
    if method not in METHODS:
        raise ValueError(f"Unknown ranking method: {method}")
    if not candidates.item_id.tolist() == features.item_id.tolist() == source_scores.item_id.tolist():
        raise ValueError("Candidate, feature, and source-score rows must align")
    if method == "logistic":
        if ranker is None:
            raise ValueError("Logistic model is unavailable; choose blend explicitly")
        scores = ranker.score(features)
    elif method == "blend":
        scores = blend_scores(source_scores)
    else:
        column = {"popularity": "popularity_score", "collaborative": "collab_score",
                  "content": "content_score"}[method]
        scores = source_scores[column].to_numpy(dtype=float)
    result = candidates.copy().assign(score=scores, source=method)
    return result.sort_values(["score", "item_id"], ascending=[False, True]).reset_index(drop=True)


@dataclass
class ModelBundle:
    selected_model: str
    ranker: Ranker | None
    collaborative: object = field(repr=False)
    content: object = field(repr=False)
    metadata: dict
    schema_version: int = BUNDLE_VERSION
    feature_columns: tuple = FEATURE_COLUMNS

    def validate(self, expected_model=None):
        if type(self.schema_version) is not int or self.schema_version != BUNDLE_VERSION:
            raise ValueError("Unsupported bundle version")
        if tuple(self.feature_columns) != FEATURE_COLUMNS:
            raise ValueError("Unsupported bundle feature schema")
        if self.selected_model not in ("logistic", "blend"):
            raise ValueError("Unsupported configured model")
        if expected_model is not None and expected_model != self.selected_model:
            raise ValueError("Artifact does not match the requested model choice")
        if self.selected_model == "logistic" and self.ranker is None:
            raise ValueError("Configured logistic model is missing")
        if self.ranker is not None:
            if not isinstance(self.ranker, Ranker) or tuple(self.ranker.feature_columns) != FEATURE_COLUMNS:
                raise ValueError("Invalid saved ranker schema")
            pipeline = self.ranker.pipeline
            if tuple(pipeline.feature_names_in_) != FEATURE_COLUMNS:
                raise ValueError("Pipeline does not match saved features")
            if pipeline.named_steps["model"].classes_.tolist() != [0, 1]:
                raise ValueError("Invalid saved classifier classes")
        cutoff = as_utc(self.metadata["serving_cutoff"])
        if any(as_utc(model.as_of_time) != cutoff for model in (self.collaborative, self.content)):
            raise ValueError("Generator cutoff does not match bundle cutoff")
        if (set(self.collaborative.item_ids) != set(self.content.item_ids)
                or list(self.collaborative.item_ids) != self.metadata["item_ids"]):
            raise ValueError("Generator catalogs do not match bundle catalog")
        if (self.metadata["candidate_budgets"] != CANDIDATE_BUDGETS
                or self.metadata["event_weights"] != EVENT_WEIGHTS
                or self.metadata["blend_weights"] != BLEND_WEIGHTS
                or self.metadata["feature_columns"] != list(FEATURE_COLUMNS)):
            raise ValueError("Bundle configuration does not match this implementation")
        training = self.metadata["training_snapshots"]
        if not training or any(as_utc(s["outcome_end"]) > cutoff
                               or as_utc(s["snapshot"]) >= cutoff for s in training):
            raise ValueError("Training windows must precede validation")
        if self.metadata.get("test_reserved") is not True:
            raise ValueError("Bundle must preserve the final test window")
        return self

    def recommend(self, visitor_id, history, context):
        """Direct Python path for Phase 7; context/history are supplied at startup."""
        if context.as_of_time != self.collaborative.as_of_time:
            raise ValueError("Feature context cutoff does not match bundle")
        if set(context.catalog.index) != set(self.collaborative.item_ids):
            raise ValueError("Feature context catalog does not match bundle")
        expected = context.histories.get(visitor_id, context.history.iloc[:0])
        pd.testing.assert_frame_equal(history.reset_index(drop=True), expected.reset_index(drop=True))
        candidates = generate_candidates(visitor_id, history, self.collaborative, self.content)
        features = build_rank_features(visitor_id, candidates, context.as_of_time, context)
        # Missing-history features were not represented in the training cohort.
        method = "popularity" if history.empty else self.selected_model
        if method == "logistic":
            sources = pd.DataFrame({"item_id": candidates.item_id})
        else:
            sources = full_source_scores(candidates, visitor_id, history, self.collaborative, self.content)
        return rank_pool(method, candidates, features, sources, self.ranker)


def save_bundle(bundle, path):
    """Atomically save a validated bundle; only load trusted local pickle files."""
    bundle.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_bundle(path, expected_model):
    """Load an explicitly configured model; missing/corrupt files never fall back."""
    try:
        with Path(path).open("rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, ModelBundle):
            raise ValueError("Artifact is not a model bundle")
        return bundle.validate(expected_model)
    except (EOFError, pickle.UnpicklingError, AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Malformed model bundle") from exc
