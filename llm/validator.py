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
)


def validate_query_plan(raw_plan: dict[str, Any]) -> QueryPlan:
    """Repair small omissions, then validate a raw LLM QueryPlan dict.

    Day 1 intentionally avoids guessing invalid task types. If ``task_type`` is
    missing or not one of the V1 enum values, this function raises ValueError.
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

    for field_name in LIST_FIELDS:
        if plan.get(field_name) is None:
            plan[field_name] = []

    if plan.get("confidence") is None:
        plan["confidence"] = 0.5
    else:
        try:
            confidence = float(plan["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be a number between 0 and 1") from exc
        plan["confidence"] = min(1.0, max(0.0, confidence))

    try:
        return QueryPlan(**plan)
    except ValidationError as exc:
        raise ValueError(f"Invalid QueryPlan payload: {exc}") from exc
