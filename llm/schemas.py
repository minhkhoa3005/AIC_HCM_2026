"""Pydantic schemas for the Day 1 LLM/retrieval contract."""

from typing import Any

from pydantic import BaseModel, Field

from .task_types import TaskType


class EntityAction(BaseModel):
    """An action attached to an entity, optionally pointing at a target."""

    verb: str
    target: str | None = None


class Entity(BaseModel):
    """A person, object, vehicle, location, or unknown entity in a query."""

    name: str
    type: str = "unknown"
    attributes: list[str] = Field(default_factory=list)
    actions: list[EntityAction] = Field(default_factory=list)


class TrakeEvent(BaseModel):
    """One ordered event milestone for a TRAKE query."""

    event_id: int
    description: str


class CaptureContext(BaseModel):
    """Optional camera/source context for visual CLIP query expansion."""

    source: str | None = None
    camera_style: str | None = None
    viewpoint: str | None = None
    camera_motion: str | None = None
    certainty: str = "unknown"


class VisualHints(BaseModel):
    """Structured visual hints used to make CLIP queries more specific."""

    # The dataset modality is known even when the user does not mention it.
    medium: str | None = "video frame"
    capture_context: CaptureContext | None = None
    core_subjects: list[str] = Field(default_factory=list)
    hypernyms: dict[str, list[str]] = Field(default_factory=dict)
    states: list[str] = Field(default_factory=list)
    environment: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Normalized LLM output used by downstream retrieval and reranking."""

    raw_query: str = ""
    rewritten_query: str = ""
    task_type: TaskType
    search_description: str
    question: str | None = None
    events: list[TrakeEvent] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scene: list[str] = Field(default_factory=list)
    positive_constraints: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    metadata_keywords: list[str] = Field(default_factory=list)
    visual_hints: VisualHints = Field(default_factory=VisualHints)
    clip_queries: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class Candidate(BaseModel):
    """Retrieval candidate returned to the LLM/reranker layer."""

    video_id: str
    frame_id: int
    keyframe_path: str
    clip_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    objects: list[str] = Field(default_factory=list)
    ocr: str = ""
    asr: str = ""
