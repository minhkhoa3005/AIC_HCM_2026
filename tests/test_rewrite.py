import json
from types import SimpleNamespace

import pytest

from llm.config import LLMConfig
from llm.rewrite import rewrite_query_with_llm, validate_rewrite_result


class MockLLMClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False))


def _config():
    return LLMConfig(
        llm_provider="gemini",
        llm_api_keys=("test-key",),
        llm_model="gemini-2.5-flash",
        llm_rewrite_temperature=0,
        llm_planner_temperature=0,
    )


def test_rewrite_returns_accented_vietnamese_and_keeps_raw_query():
    raw_query = "Tim canh ng dan ong mac ao do mo cua xe trang"
    client = MockLLMClient(
        {"rewritten_query": "Tìm cảnh người đàn ông mặc áo đỏ mở cửa xe trắng"}
    )

    result = rewrite_query_with_llm(raw_query, config=_config(), client=client)

    assert result.raw_query == raw_query
    assert result.rewritten_query == "Tìm cảnh người đàn ông mặc áo đỏ mở cửa xe trắng"
    assert "tiếng Việt có đầy đủ dấu" in client.calls[0]["contents"]


def test_rewrite_rejects_empty_rewrite():
    with pytest.raises(ValueError, match="rewritten_query"):
        validate_rewrite_result({"rewritten_query": "  "})


def test_rewrite_rejects_empty_query():
    with pytest.raises(ValueError, match="non-empty string"):
        rewrite_query_with_llm("  ", config=_config(), client=MockLLMClient({}))
