"""Lightweight validation and repair helpers for LLM JSON output."""

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .schemas import QueryPlan
from .task_types import TaskType

LIST_FIELDS = (
    "events",
    "entities",
    "objects",
    "actions",
    "scene",
    "positive_constraints",
    "negative_constraints",
    "metadata_keywords",
    "clip_queries",
)

VISUAL_LIST_FIELDS = (
    "core_subjects",
    "states",
    "environment",
)


def validate_query_plan(raw_plan: dict[str, Any]) -> QueryPlan:
    """Repair common LLM JSON shape deviations, then validate a QueryPlan.

    Invalid or missing ``task_type`` values raise ValueError instead of being
    guessed locally.
    """

    if not isinstance(raw_plan, dict):
        raise TypeError("QueryPlan payload must be a dict")

    plan = deepcopy(raw_plan)
    task_type = plan.get("task_type")
    valid_task_types = {item.value for item in TaskType}
    if task_type not in valid_task_types:
        raise ValueError(
            f"Invalid task_type {task_type!r}. Expected one of: "
            f"{', '.join(sorted(valid_task_types))}"
        )

    _repair_list_fields(plan, LIST_FIELDS)

    if plan.get("visual_hints") is None:
        plan["visual_hints"] = {}
    _repair_visual_hints(plan)

    try:
        validated = QueryPlan(**plan)
    except ValidationError as exc:
        raise ValueError(f"Invalid QueryPlan payload: {exc}") from exc

    _validate_task_invariants(validated)
    return validated


def _repair_list_fields(payload: dict[str, Any], field_names: tuple[str, ...]) -> None:
    """Convert null or one scalar list item into the schema's list shape."""

    for field_name in field_names:
        value = payload.get(field_name)
        if value is None:
            payload[field_name] = []
        elif isinstance(value, str):
            payload[field_name] = [value]


def _repair_visual_hints(plan: dict[str, Any]) -> None:
    """Normalize nullable/narrow visual-hint fields emitted by the planner."""

    visual_hints = plan.get("visual_hints")
    if not isinstance(visual_hints, dict):
        return

    _repair_list_fields(visual_hints, VISUAL_LIST_FIELDS)

    hypernyms = visual_hints.get("hypernyms")
    if hypernyms is None:
        visual_hints["hypernyms"] = {}
    elif isinstance(hypernyms, dict):
        for name, values in hypernyms.items():
            if values is None:
                hypernyms[name] = []
            elif isinstance(values, str):
                hypernyms[name] = [values]


def _validate_task_invariants(plan: QueryPlan) -> None:
    """Reject task payloads whose fields cannot be routed safely downstream."""

    if plan.task_type == TaskType.TEXTUAL_KIS:
        if plan.question is not None:
            raise ValueError("TEXTUAL_KIS plans must set question to null")
        if plan.events:
            raise ValueError("TEXTUAL_KIS plans must not contain events")
        return

    if plan.task_type == TaskType.QA:
        if not plan.question or not plan.question.strip():
            raise ValueError("QA plans must contain a non-empty question")
        return

    if plan.question is not None:
        raise ValueError("TRAKE plans must set question to null")
    if not plan.events:
        raise ValueError("TRAKE plans must contain ordered events")

    event_ids = [event.event_id for event in plan.events]
    expected_ids = list(range(1, len(plan.events) + 1))
    if event_ids != expected_ids:
        raise ValueError("TRAKE event_id values must be consecutive and start at 1")
    if len(plan.clip_queries) != len(plan.events):
        raise ValueError("TRAKE plans must contain exactly one clip query per event")
