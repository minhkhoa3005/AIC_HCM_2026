"""Stable internal result object for downstream submission export."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm.task_types import TaskType


class Prediction(BaseModel):
    query_id: str
    task_type: TaskType
    rank: int
    video_id: str
    frame_ids: list[int] = Field(default_factory=list)
    answer: str | None = None
    score: float | None = None
