"""CSV exporter for final predictions."""

from __future__ import annotations

import csv
from pathlib import Path

from pipeline.prediction import Prediction

from .config import SubmissionConfig
from .validation import validate_predictions


def export_csv(
    predictions: list[Prediction],
    path: str | Path,
    config: SubmissionConfig | None = None,
) -> Path:
    """Write predictions to CSV. This is the only module that formats submissions."""

    resolved_config = config or SubmissionConfig()
    validate_predictions(predictions)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=resolved_config.separator)
        writer.writerow(resolved_config.header)
        for prediction in predictions:
            writer.writerow(_row(prediction, resolved_config))
    return output_path


def _row(prediction: Prediction, config: SubmissionConfig) -> list[str | int]:
    video_id = prediction.video_id
    if config.include_mp4_extension and not video_id.endswith(".mp4"):
        video_id = f"{video_id}.mp4"
    frame_ids = [frame_id + config.frame_base for frame_id in prediction.frame_ids]
    return [
        prediction.query_id,
        prediction.rank,
        video_id,
        " ".join(str(frame_id) for frame_id in frame_ids),
        prediction.answer or "",
    ]
