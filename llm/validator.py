"""Lightweight validation and repair helpers for LLM JSON output."""

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .schemas import QueryPlan
from .task_types import TaskType

LIST_FIELDS = (
    "entities",
    "objects",
    "actions",
    "scene",
    "positive_constraints",
    "negative_constraints",
    "metadata_keywords",
    "expansions",
    "clip_queries",
    "events",
    "event_queries",
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
    if validated.task_type == TaskType.TRAKE:
        return validated
    if not validated.anchor.strip():
        raise ValueError("QueryPlan.anchor must contain an English CLIP anchor")
    if len(validated.expansions) != 10:
        raise ValueError("QueryPlan.expansions must contain exactly 10 English expansions")
    english_fields = [validated.anchor, *validated.expansions]
    if any(not isinstance(value, str) or not value.strip() for value in english_fields):
        raise ValueError("anchor and expansions must be non-empty strings")
    if any(not value.isascii() for value in english_fields):
        raise ValueError("anchor and expansions must contain English ASCII text")
    if len(set(value.casefold() for value in validated.expansions)) != 10:
        raise ValueError("expansions must be distinct")
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
        return

    if plan.task_type == TaskType.QA:
        if not plan.question or not plan.question.strip():
            raise ValueError("QA plans must contain a non-empty question")
        return

    if plan.task_type == TaskType.TRAKE:
        if len(plan.events) < 2:
            raise ValueError("TRAKE plans must contain at least two ordered events")
        if len(plan.event_queries) != len(plan.events):
            raise ValueError("TRAKE requires one English event_query per event")
        if any(not query.strip() or not query.isascii() for query in plan.event_queries):
            raise ValueError("TRAKE event_queries must be non-empty English ASCII")
        return

    raise ValueError(f"Unsupported task_type: {plan.task_type}")
