"""QA task handling over retrieved visual evidence."""

from __future__ import annotations

from llm.schemas import Candidate, QueryPlan
from llm.task_types import TaskType

from .prediction import Prediction


def answer_qa(query_id: str, plan: QueryPlan, candidates: list[Candidate]) -> list[Prediction]:
    """Create QA predictions with a deterministic evidence fallback."""

    if not candidates:
        return []
    candidate = candidates[0]
    answer = _fallback_answer(candidate)
    return [
        Prediction(
            query_id=query_id,
            task_type=TaskType.QA,
            rank=1,
            video_id=candidate.video_id,
            frame_ids=[candidate.frame_id],
            answer=answer,
            score=candidate.clip_score,
        )
    ]


def _fallback_answer(candidate: Candidate) -> str:
    if candidate.ocr:
        return candidate.ocr
    if candidate.asr:
        return candidate.asr
    return ""
