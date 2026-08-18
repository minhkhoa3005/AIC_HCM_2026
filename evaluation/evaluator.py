"""Local evaluator following the AIC Top-1/5/20/50/100 scoring rule."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from pipeline.prediction import Prediction


@dataclass(frozen=True)
class GroundTruth:
    """One accepted answer region for a query.

    ``frame_ids`` keeps compatibility with the old evaluator. For BTC answer
    ranges, use ``start_frame`` and ``end_frame`` inclusively.
    """

    video_id: str
    frame_ids: frozenset[int] = frozenset()
    answer: str | None = None
    start_frame: int | None = None
    end_frame: int | None = None

    def accepts_frame(self, frame_id: int) -> bool:
        if self.frame_ids and frame_id in self.frame_ids:
            return True
        if self.start_frame is not None and self.end_frame is not None:
            return self.start_frame <= frame_id <= self.end_frame
        return not self.frame_ids and self.start_frame is None and self.end_frame is None


AnswerMatcher = Callable[[str, str], bool]
RScoreFn = Callable[[Prediction, tuple[GroundTruth, ...]], float]


def evaluate_predictions(
    predictions: Iterable[Prediction],
    ground_truth: Mapping[str, Iterable[GroundTruth]],
    ks: tuple[int, ...] = (1, 5, 20, 50, 100),
    *,
    answer_matcher: AnswerMatcher | None = None,
    r_score_fn: RScoreFn | None = None,
) -> dict[str, object]:
    """Compute BTC-style per-query max R-score and aggregate Final Score.

    For every query, ``Top-k`` is the maximum R-score among predictions ranked
    1..k. The reported ``R@k`` is the mean of those per-query Top-k values.
    This is intentionally different from binary recall, which only asks whether
    any hit exists in the prefix.
    """
    grouped: dict[str, list[Prediction]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.query_id, []).append(prediction)
    expected_by_query = {
        query_id: tuple(items) for query_id, items in ground_truth.items()
    }
    query_ids = list(expected_by_query)
    per_query: dict[str, dict[str, float]] = {}
    aggregate: dict[str, float] = {}

    for query_id in query_ids:
        ranked = sorted(grouped.get(query_id, []), key=lambda item: item.rank)
        expected = expected_by_query[query_id]
        scores = [
            _r_score(item, expected, answer_matcher, r_score_fn)
            for item in ranked[: max(ks, default=0)]
        ]
        query_metrics: dict[str, float] = {}
        for k in ks:
            value = max(scores[:k], default=0.0)
            query_metrics[f"R@{k}"] = value
            aggregate[f"R@{k}"] = aggregate.get(f"R@{k}", 0.0) + value
        per_query[query_id] = query_metrics

    divisor = len(query_ids) or 1
    aggregate = {key: value / divisor for key, value in aggregate.items()}
    final_score = (
        sum(aggregate.values()) / len(aggregate)
        if aggregate
        else 0.0
    )
    diagnostics = [_diagnostic(query_id, grouped.get(query_id, []), per_query[query_id], expected_by_query[query_id], answer_matcher, r_score_fn) for query_id in query_ids]
    return {
        **aggregate,
        "Final Score": final_score,
        "per_query": per_query,
        "diagnostics": diagnostics,
    }


def _r_score(
    prediction: Prediction,
    expected: tuple[GroundTruth, ...],
    answer_matcher: AnswerMatcher | None,
    r_score_fn: RScoreFn | None,
) -> float:
    if r_score_fn is not None:
        return max(0.0, min(1.0, float(r_score_fn(prediction, expected))))
    return 1.0 if _prediction_hit(prediction, expected, answer_matcher) else 0.0


def _prediction_hit(
    prediction: Prediction,
    expected: tuple[GroundTruth, ...],
    answer_matcher: AnswerMatcher | None,
) -> bool:
    for item in expected:
        if prediction.video_id != item.video_id:
            continue
        if not any(item.accepts_frame(frame_id) for frame_id in prediction.frame_ids):
            continue
        if item.answer is None:
            return True
        matcher = answer_matcher or _default_answer_matcher
        if matcher(prediction.answer or "", item.answer):
            return True
    return False


def _default_answer_matcher(predicted: str, expected: str) -> bool:
    """Small local semantic-tolerant matcher; replace with official judge if available."""
    left = _normalize(predicted)
    right = _normalize(expected)
    if not left or not right:
        return left == right
    if left == right or left in right or right in left:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    overlap = len(left_tokens & right_tokens) / max(len(right_tokens), 1)
    return overlap >= 0.8


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.casefold())
    without_marks = "".join(
        character for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(re.findall(r"[\w]+", without_marks, flags=re.UNICODE))


def _diagnostic(
    query_id: str,
    predictions: list[Prediction],
    metrics: dict[str, float],
    expected: tuple[GroundTruth, ...],
    answer_matcher: AnswerMatcher | None,
    r_score_fn: RScoreFn | None,
) -> dict[str, object]:
    first_hit = None
    first_score = 0.0
    for prediction in sorted(predictions, key=lambda item: item.rank):
        score = _r_score(prediction, expected, answer_matcher, r_score_fn)
        if score > 0:
            first_hit = prediction.rank
            first_score = score
            break
    return {
        "query_id": query_id,
        "first_hit_rank": first_hit,
        "first_hit_score": first_score,
        "top_prediction": (
            {
                "video_id": predictions[0].video_id,
                "frame_ids": predictions[0].frame_ids,
                "answer": predictions[0].answer,
            }
            if predictions
            else None
        ),
        "expected_videos": sorted({item.video_id for item in expected}),
        "metrics": metrics,
        "status": "hit" if first_hit is not None else "miss",
    }
