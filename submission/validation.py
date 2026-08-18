"""Pre-export validation for the organizer's maximum-100 format."""

from __future__ import annotations

from collections import defaultdict

from pipeline.prediction import Prediction


def validate_predictions(
    predictions: list[Prediction],
    *,
    max_per_query: int = 100,
    require_qa_answers: bool = False,
) -> None:
    """Reject malformed ranked output before writing a submission CSV."""
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.query_id].append(prediction)
    for query_id, rows in grouped.items():
        rows.sort(key=lambda item: item.rank)
        if len(rows) > max_per_query:
            raise ValueError(f"{query_id} has {len(rows)} predictions; maximum is {max_per_query}")
        expected_ranks = list(range(1, len(rows) + 1))
        actual_ranks = [row.rank for row in rows]
        if actual_ranks != expected_ranks:
            raise ValueError(f"{query_id} ranks must be contiguous from 1: {actual_ranks}")
        keys = [(row.video_id, tuple(row.frame_ids)) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{query_id} contains duplicate video/frame predictions")
        if require_qa_answers and any(row.answer is None or not row.answer.strip() for row in rows):
            raise ValueError(f"{query_id} contains a QA prediction without an answer")
