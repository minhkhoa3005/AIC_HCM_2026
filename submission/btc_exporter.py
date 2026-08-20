"""Export one BTC submission CSV per query without a header row."""

from __future__ import annotations

import csv
from pathlib import Path

from llm.task_types import TaskType, normalize_task_type
from pipeline.prediction import Prediction

from .validation import validate_predictions


def export_btc_csv(
    predictions: list[Prediction],
    path: str | Path,
    *,
    task_type: TaskType | str,
) -> Path:
    """Write the organizer's task-specific, headerless CSV format.

    KIS rows are ``video_id,frame_id``; QA rows add the answer; TRAKE rows
    contain the ordered event frame IDs after the video ID.
    """

    resolved_task = normalize_task_type(task_type)
    validate_predictions(
        predictions,
        require_qa_answers=resolved_task == TaskType.QA,
    )
    for prediction in predictions:
        if prediction.task_type != resolved_task:
            raise ValueError(
                f"Prediction task type {prediction.task_type} does not match {resolved_task}"
            )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=",")
        for prediction in sorted(predictions, key=lambda item: item.rank):
            if resolved_task in {TaskType.TEXTUAL_KIS, TaskType.QA}:
                if len(prediction.frame_ids) != 1:
                    raise ValueError(
                        f"{resolved_task.value} prediction must contain exactly one frame_id"
                    )
                row: list[str | int] = [prediction.video_id, prediction.frame_ids[0]]
                if resolved_task == TaskType.QA:
                    row.append(prediction.answer or "")
            else:
                if len(prediction.frame_ids) < 2:
                    raise ValueError("TRAKE prediction must contain at least two event frame IDs")
                row = [prediction.video_id, *prediction.frame_ids]
            writer.writerow(row)
    return output_path
