import json

import pytest

from llm.classifier import classify_intent, validate_intent_classification
from llm.examples import DAY1_QUERY_EXAMPLES
from llm.schemas import IntentClassification
from llm.task_types import TaskType


def test_classifier_handles_representative_textual_kis_query():
    result = classify_intent("Tim canh nguoi dan ong mac ao do mo cua xe mau trang.")

    assert result.task_type == TaskType.TEXTUAL_KIS
    assert result.confidence >= 0.5
    assert result.reason


def test_classifier_handles_representative_qa_query():
    result = classify_intent("Trong canh nguoi phu nu dung truoc xe buyt, bien so xe la gi?")

    assert result.task_type == TaskType.QA
    assert result.confidence >= 0.5


def test_classifier_handles_representative_trake_query():
    result = classify_intent(
        "Tim chuoi su kien nguoi mo cua xe, buoc ra ngoai, roi di vao toa nha."
    )

    assert result.task_type == TaskType.TRAKE
    assert result.confidence >= 0.5


@pytest.mark.parametrize("sample", DAY1_QUERY_EXAMPLES)
def test_classifier_matches_each_day1_example(sample):
    predicted = classify_intent(sample["input"])

    assert predicted.task_type.value == sample["expected_task_type"]


def test_intent_classification_json_round_trip():
    result = classify_intent("Nguoi dan ong dang cam vat gi khi buoc vao cua hang?")
    recreated = IntentClassification(**json.loads(result.model_dump_json()))

    assert recreated.task_type == TaskType.QA
    assert recreated.reason


def test_validate_intent_classification_clamps_confidence():
    high = validate_intent_classification(
        {"task_type": "TEXTUAL_KIS", "confidence": 2, "reason": "test"}
    )
    low = validate_intent_classification(
        {"task_type": "TEXTUAL_KIS", "confidence": -1, "reason": "test"}
    )

    assert high.confidence == 1.0
    assert low.confidence == 0.0


def test_validate_intent_classification_rejects_invalid_task_type():
    with pytest.raises(ValueError, match="Invalid task_type"):
        validate_intent_classification({"task_type": "SEARCH", "confidence": 0.9})


def test_classifier_rejects_empty_query():
    with pytest.raises(ValueError, match="non-empty string"):
        classify_intent("   ")
