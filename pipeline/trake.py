"""TRAKE temporal alignment over per-event retrieval results."""

from __future__ import annotations

from collections import defaultdict

from llm.schemas import Candidate, QueryPlan
from llm.task_types import TaskType

from .prediction import Prediction


def align_trake(query_id: str, plan: QueryPlan, candidates: list[Candidate]) -> list[Prediction]:
    """Group by video and keep increasing frame order for ordered events."""

    if not candidates:
        return []
    by_video: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_video[candidate.video_id].append(candidate)

    predictions: list[Prediction] = []
    for video_id, video_candidates in by_video.items():
        ordered = sorted(video_candidates, key=lambda item: item.frame_id)
        frame_ids = [candidate.frame_id for candidate in ordered[: max(1, len(plan.events))]]
        if frame_ids != sorted(frame_ids):
            continue
        predictions.append(
            Prediction(
                query_id=query_id,
                task_type=TaskType.TRAKE,
                rank=len(predictions) + 1,
                video_id=video_id,
                frame_ids=frame_ids,
                score=sum(candidate.clip_score for candidate in ordered) / len(ordered),
            )
        )
    return predictions
