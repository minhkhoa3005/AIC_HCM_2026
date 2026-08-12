"""LLM-backed parser that returns the QueryPlan contract."""

from __future__ import annotations

import json
from typing import Any

from ..classifier import classify_intent
from ..config import LLMConfig, get_llm_config
from ..llm_client import generate_llm_json
from ..prompts import QUERY_PLAN_PARSER_PROMPT
from ..schemas import IntentClassification, QueryPlan
from ..validator import validate_query_plan


def parse_query_plan_with_llm(
    query: str,
    intent: IntentClassification | None = None,
    config: LLMConfig | None = None,
    client: Any | None = None,
) -> QueryPlan:
    """Parse fields with an LLM while preserving the classified task type."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    resolved_config = config or get_llm_config()
    resolved_intent = intent or classify_intent(query)
    prompt = QUERY_PLAN_PARSER_PROMPT.format(
        query=query,
        intent=json.dumps(resolved_intent.model_dump(mode="json"), ensure_ascii=False),
    )

    raw_plan = generate_llm_json(
        prompt,
        config=resolved_config,
        temperature=resolved_config.llm_parser_temperature,
        client=client,
    )
    raw_plan["task_type"] = resolved_intent.task_type.value
    return validate_query_plan(raw_plan)
