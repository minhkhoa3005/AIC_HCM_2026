"""Validation-set calibration for weighted reciprocal-rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from llm.schemas import QueryPlan
from pipeline.rerank import DEFAULT_RRF_WEIGHTS, fuse_retrieval_results, rerank_candidates
from retrieval.models import QueryRetrievalResult

from .evaluator import GroundTruth


@dataclass(frozen=True)
class RRFValidationCase:
    query_id: str
    plan: QueryPlan
    results: list[QueryRetrievalResult]
    ground_truth: tuple[GroundTruth, ...]


@dataclass(frozen=True)
class RRFCalibration:
    weights: tuple[float, float, float]
    score: float
    evaluated_cases: int


def calibrate_rrf_weights(
    cases: list[RRFValidationCase],
    *,
    weight_grid: tuple[tuple[float, ...], ...] | None = None,
    shortlist_limit: int = 300,
) -> RRFCalibration:
    """Choose RRF weights maximizing frame/video hit rate on validation data."""
    if not cases:
        return RRFCalibration(DEFAULT_RRF_WEIGHTS, 0.0, 0)
    grid = weight_grid or (
        (1.0, 0.85, 0.7),
        (1.0, 0.9, 0.8),
        (1.0, 0.8, 0.6),
        (1.0, 0.7, 0.7),
        (1.0, 0.6, 0.4),
    )
    best = RRFCalibration(DEFAULT_RRF_WEIGHTS, -1.0, len(cases))
    for weights in grid:
        if len(weights) != 3 or any(weight < 0 for weight in weights):
            raise ValueError("Each RRF weight candidate must contain three non-negative values")
        case_scores: list[float] = []
        for case in cases:
            fused = fuse_retrieval_results(
                case.results,
                case.plan,
                query_weights=list(weights),
                limit=shortlist_limit,
            )
            ranked = rerank_candidates(fused, case.plan, limit=100)
            top_scores = [
                1.0 if _matches(item.candidate.video_id, item.candidate.frame_id, case.ground_truth) else 0.0
                for item in ranked
            ]
            top_k_scores = [max(top_scores[:k], default=0.0) for k in (1, 5, 20, 50, 100)]
            case_scores.append(sum(top_k_scores) / len(top_k_scores))
        score = sum(case_scores) / len(case_scores)
        if score > best.score:
            best = RRFCalibration(tuple(weights), score, len(cases))
    return best


def _matches(video_id: str, frame_id: int, expected: tuple[GroundTruth, ...]) -> bool:
    return any(
        video_id == item.video_id and item.accepts_frame(frame_id)
        for item in expected
    )
