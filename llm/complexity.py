"""Complexity gate for deciding when an LLM parser is worth using."""

from __future__ import annotations

from .classifier import normalize_text
from .config import LLMConfig, get_llm_config
from .schemas import IntentClassification

TEMPORAL_CONNECTORS = (
    "roi",
    "sau do",
    "tiep theo",
    "truoc khi",
    "sau khi",
    "lan luot",
    "theo thu tu",
)

ACTION_SIGNALS = (
    "mo",
    "dong",
    "buoc",
    "di",
    "chay",
    "cam",
    "lay",
    "dua",
    "dat",
    "quet",
    "cat",
    "chup",
    "dung",
    "ngoi",
    "xep",
)

ENTITY_SIGNALS = (
    "nguoi",
    "xe",
    "cua",
    "bien",
    "logo",
    "man hinh",
    "dien thoai",
    "laptop",
    "tui",
    "hang",
    "toa nha",
    "cua hang",
    "quan",
)


def is_complex_query(
    query: str,
    intent: IntentClassification,
    confidence_threshold: float = 0.8,
) -> bool:
    """Return True when a query likely benefits from semantic LLM parsing."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    normalized = normalize_text(query)
    words = normalized.split()
    temporal_count = sum(
        1 for connector in TEMPORAL_CONNECTORS if _contains_phrase(normalized, connector)
    )
    action_count = sum(1 for signal in ACTION_SIGNALS if _contains_phrase(normalized, signal))
    entity_count = sum(1 for signal in ENTITY_SIGNALS if _contains_phrase(normalized, signal))

    return any(
        (
            len(words) >= 18,
            normalized.count(",") >= 2,
            temporal_count >= 2,
            temporal_count >= 1 and action_count >= 2,
            action_count >= 3 and entity_count >= 2,
            intent.confidence < confidence_threshold,
        )
    )


def should_use_llm_parser(
    query: str,
    intent: IntentClassification,
    config: LLMConfig | None = None,
) -> bool:
    """Apply env flags and complexity mode to decide parser routing."""

    resolved_config = config or get_llm_config()
    if not resolved_config.use_llm_parser:
        return False

    mode = resolved_config.llm_parser_mode
    if mode == "always":
        return True
    if mode == "auto":
        return is_complex_query(
            query,
            intent,
            confidence_threshold=resolved_config.intent_confidence_threshold,
        )
    if mode == "never":
        return False
    raise ValueError(
        f"Invalid LLM_PARSER_MODE {mode!r}. Expected one of: auto, always, never"
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "
