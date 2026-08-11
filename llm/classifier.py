"""Intent classification for Day 2.

The classifier is deterministic first. Gemini/API classification can reuse the
same output schema and fall back to this module when confidence is high enough.
"""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .schemas import IntentClassification
from .task_types import TaskType

QUESTION_KEYWORDS = (
    "bao nhieu",
    "mau gi",
    "la gi",
    "cai gi",
    "vat gi",
    "chu gi",
    "so gi",
    "ai",
    "o dau",
    "nhu the nao",
    "co khong",
    "hay khong",
    "ben trai hay ben phai",
    "bang tay nao",
    "phuong tien nao",
    "loai gi",
)

TRAKE_KEYWORDS = (
    "tim chuoi",
    "tim doan",
    "chuoi su kien",
    "theo thu tu",
    "lan luot",
)

TEMPORAL_CONNECTORS = (
    "roi",
    "sau do",
    "tiep theo",
    "truoc khi",
    "sau khi",
)

TEXTUAL_SEARCH_KEYWORDS = (
    "tim canh",
    "tim khoanh khac",
    "tim hinh anh",
    "tim khung hinh",
)

INTENT_FIELDS = ("task_type", "confidence", "reason")


def normalize_text(text: str) -> str:
    """Lowercase Vietnamese text and remove accents for keyword matching."""

    lowered = text.lower().strip()
    decomposed = unicodedata.normalize("NFD", lowered)
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    without_vietnamese_d = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_vietnamese_d)


def _contains_keyword(text: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def validate_intent_classification(raw_result: dict[str, Any]) -> IntentClassification:
    """Validate and lightly repair classifier output.

    Day 2 does not guess invalid task types. Missing confidence defaults to 0.5,
    and numeric confidence is clamped to [0, 1].
    """

    if not isinstance(raw_result, dict):
        raise TypeError("Intent classification payload must be a dict")

    result = deepcopy(raw_result)
    valid_task_types = {item.value for item in TaskType}
    if result.get("task_type") not in valid_task_types:
        raise ValueError(
            f"Invalid task_type {result.get('task_type')!r}. Expected one of: "
            f"{', '.join(sorted(valid_task_types))}"
        )

    if result.get("confidence") is None:
        result["confidence"] = 0.5
    else:
        try:
            confidence = float(result["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be a number between 0 and 1") from exc
        result["confidence"] = min(1.0, max(0.0, confidence))

    if result.get("reason") is None:
        result["reason"] = ""

    try:
        return IntentClassification(
            **{key: result.get(key) for key in INTENT_FIELDS}
        )
    except ValidationError as exc:
        raise ValueError(f"Invalid IntentClassification payload: {exc}") from exc


def classify_intent(query: str) -> IntentClassification:
    """Classify a user query into TEXTUAL_KIS, QA, or TRAKE."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    normalized = normalize_text(query)
    has_question_mark = "?" in query
    has_question_keyword = any(
        _contains_keyword(normalized, keyword) for keyword in QUESTION_KEYWORDS
    )
    has_trake_keyword = any(
        _contains_keyword(normalized, keyword) for keyword in TRAKE_KEYWORDS
    )
    has_temporal_connector = any(
        _contains_keyword(normalized, connector) for connector in TEMPORAL_CONNECTORS
    )
    comma_count = normalized.count(",")
    has_search_keyword = any(keyword in normalized for keyword in TEXTUAL_SEARCH_KEYWORDS)

    if has_trake_keyword and (has_temporal_connector or comma_count >= 1):
        return IntentClassification(
            task_type=TaskType.TRAKE,
            confidence=0.92,
            reason="Query asks for an ordered sequence of events.",
        )

    if has_temporal_connector and comma_count >= 1:
        return IntentClassification(
            task_type=TaskType.TRAKE,
            confidence=0.78,
            reason="Query contains multiple temporal actions.",
        )

    if has_search_keyword and comma_count >= 2:
        return IntentClassification(
            task_type=TaskType.TRAKE,
            confidence=0.74,
            reason="Query contains a multi-step visual sequence.",
        )

    if has_question_mark or has_question_keyword:
        return IntentClassification(
            task_type=TaskType.QA,
            confidence=0.88,
            reason="Query asks for an answer after locating visual evidence.",
        )

    if has_search_keyword:
        return IntentClassification(
            task_type=TaskType.TEXTUAL_KIS,
            confidence=0.86,
            reason="Query asks to find a visual moment from description.",
        )

    return IntentClassification(
        task_type=TaskType.TEXTUAL_KIS,
        confidence=0.55,
        reason="Fallback to visual search because no QA or TRAKE signal was found.",
    )
