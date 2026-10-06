"""HTTP response contracts; scores are ordering signals, not probabilities."""

from typing import Literal

from pydantic import BaseModel, FiniteFloat


class Recommendation(BaseModel):
    item_id: str
    score: FiniteFloat
    sources: list[Literal["collaborative", "content", "popularity"]]


class RecommendationResponse(BaseModel):
    visitor_id: str
    model_version: str
    recommendations: list[Recommendation]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    model: Literal["logistic", "blend"]
    serving_cutoff: str
