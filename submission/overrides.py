"""Manual prediction overrides before CSV regeneration."""

from __future__ import annotations

from pipeline.prediction import Prediction


def apply_overrides(
    predictions: list[Prediction],
    overrides: dict[str, list[Prediction]],
) -> list[Prediction]:
    """Replace predictions for selected query IDs with curated results."""

    result: list[Prediction] = []
    for prediction in predictions:
        if prediction.query_id in overrides:
            continue
        result.append(prediction)
    for query_predictions in overrides.values():
        result.extend(query_predictions)
    return sorted(result, key=lambda item: (item.query_id, item.rank))
