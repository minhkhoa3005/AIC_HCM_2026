"""Regression tests for local model JSON shape normalization."""

import pytest

from llm.vlm.local import _parse_json_object


def test_accepts_json_object() -> None:
    assert _parse_json_object('{"answer": "blue"}') == {"answer": "blue"}


def test_selects_first_object_from_model_alternatives() -> None:
    payload = _parse_json_object(
        '[{"rewritten_query": "first"}, {"rewritten_query": "second"}]'
    )
    assert payload == {"rewritten_query": "first"}


def test_unwraps_single_object_wrapper() -> None:
    assert _parse_json_object('{"result": {"answer": "green"}}') == {
        "answer": "green"
    }


def test_wraps_raw_reranking_items() -> None:
    payload = _parse_json_object(
        '[{"index": 0, "video_id": "L01_V001", "frame_id": 10, '
        '"relevant": true, "confidence": 0.9}]'
    )
    assert payload["items"][0]["frame_id"] == 10


def test_recovers_fenced_json_with_explanation() -> None:
    payload = _parse_json_object(
        'Result:\n```json\n{"answer": "white"}\n```\nDone.'
    )
    assert payload == {"answer": "white"}


def test_does_not_treat_nested_array_as_top_level_payload() -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_json_object(
            '{"rewritten_query": "query", "preserved_terms": []'
        )
