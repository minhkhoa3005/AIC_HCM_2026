"""LLM-backed intent classifier."""

from __future__ import annotations

from typing import Any

from ..config import LLMConfig, get_llm_config
from ..llm_client import generate_llm_json
from ..prompts import INTENT_CLASSIFIER_PROMPT
from ..schemas import IntentClassification
from .local_classifier import validate_intent_classification


def classify_intent_with_llm(
    query: str,
    config: LLMConfig | None = None,
    client: Any | None = None,
) -> IntentClassification:
    """Classify task type with an LLM and validate the classifier contract."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    resolved_config = config or get_llm_config()
    prompt = INTENT_CLASSIFIER_PROMPT.format(query=query)
    raw_intent = generate_llm_json(
        prompt,
        config=resolved_config,
        temperature=resolved_config.llm_classifier_temperature,
        client=client,
    )
    return validate_intent_classification(raw_intent)
