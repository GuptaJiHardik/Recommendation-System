"""Local, snapshot-frozen HTTP serving of an explicitly selected model bundle."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sqlite3
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Path as PathParameter, Query, Request

from src.candidates import SOURCE_COLUMNS
from src.data import DEFAULT_DB, load_events, load_items
from src.final_ranking import apply_final_ranking
from src.rank_features import RankFeatureContext, prepare_rank_context
from src.ranker import ModelBundle, load_bundle
from src.schemas import HealthResponse, Recommendation, RecommendationResponse


DEFAULT_ARTIFACT = Path("artifacts/model_bundle.pkl")


@dataclass(frozen=True)
class ServingState:
    bundle: ModelBundle
    context: RankFeatureContext
    model_version: str
    category_share: float | None


def create_app(*, db_path=None, artifact_path=None, expected_model=None, category_share=0.4):
    """Create an app without reading data until its lifespan starts.

    Explicit arguments override environment settings. Paths are relative to the
    working directory; run from the project root. Only trusted local bundles
    should be configured because the existing artifact format is pickle.
    """
    @asynccontextmanager
    async def lifespan(app):
        database = Path(db_path if db_path is not None else os.environ.get("RECOMMENDATION_DB", DEFAULT_DB))
        artifact = Path(artifact_path if artifact_path is not None else os.environ.get(
            "RECOMMENDATION_ARTIFACT", DEFAULT_ARTIFACT))
        model = expected_model if expected_model is not None else os.environ.get("RECOMMENDATION_MODEL", "logistic")
        try:
            if model not in ("logistic", "blend"):
                raise ValueError("RECOMMENDATION_MODEL must be logistic or blend")
            bundle = load_bundle(artifact, expected_model=model)
            context = prepare_rank_context(load_events(database), load_items(database),
                                           bundle.metadata["serving_cutoff"])
            if context.as_of_time != bundle.collaborative.as_of_time:
                raise ValueError("Feature context cutoff does not match bundle")
            if set(context.catalog.index) != set(bundle.collaborative.item_ids):
                raise ValueError("Feature context catalog does not match bundle")
            # Validate final-ranking configuration and categories before readiness.
            apply_final_ranking(pd.DataFrame({"item_id": [], "score": []}),
                                context.catalog, set(), 1, category_share)
            with artifact.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()[:12]
            state = ServingState(bundle, context, f"local-{digest}", category_share)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, sqlite3.Error) as exc:
            raise RuntimeError(f"Recommendation startup failed: {exc}") from exc
        app.state.serving = state
        try:
            yield
        finally:
            app.state.serving = None

    app = FastAPI(title="Recommendation MVP", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        state = request.app.state.serving
        return HealthResponse(status="ok", model_version=state.model_version,
                              model=state.bundle.selected_model,
                              serving_cutoff=state.context.as_of_time.isoformat())

    @app.get("/recommendations/{visitor_id}", response_model=RecommendationResponse)
    def recommendations(request: Request,
                        visitor_id: Annotated[str, PathParameter(min_length=1, pattern=r"\S")],
                        k: Annotated[int, Query(ge=1, le=50)] = 10):
        state = request.app.state.serving
        context = state.context
        history = context.histories.get(visitor_id, context.history.iloc[:0])
        ranked = state.bundle.recommend(visitor_id, history, context)
        purchased = context.visitor_stats.get(visitor_id, {}).get("purchased", set())
        final = apply_final_ranking(ranked, context.catalog, purchased, k, state.category_share)
        return RecommendationResponse(
            visitor_id=visitor_id, model_version=state.model_version,
            recommendations=[Recommendation(
                item_id=row.item_id, score=float(row.score),
                sources=[source for source, (_, flag) in SOURCE_COLUMNS.items() if getattr(row, flag)]
            ) for row in final.itertuples(index=False)],
        )

    return app


app = create_app()
