"""LLM query planning: Vietnamese semantic extraction plus CLIP queries."""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import LLMConfig, get_llm_config
from .llm_client import generate_llm_json
from .rewrite import rewrite_query_with_llm
from .schemas import QueryPlan
from .task_types import TaskType, normalize_task_type
from .validator import validate_query_plan


logger = logging.getLogger(__name__)


_PLANNER_SYSTEM_PROMPT = r"""
Lập kế hoạch tìm kiếm video từ truy vấn. TASK_TYPE được cung cấp từ tên file;
không được thay đổi. Chỉ dùng chi tiết có trong truy vấn, không đoán đáp án QA,
không bịa màu sắc, người, vật, hành động hoặc bối cảnh.

Với TEXTUAL_KIS hoặc QA:
- anchor là bản dịch hình ảnh tiếng Anh ASCII sát nghĩa.
- expansions có đúng 10 mô tả hình ảnh tiếng Anh ASCII, khác nhau nhưng cùng nghĩa.
- anchor và mỗi expansion tối đa 24 từ; không kể chuyện hoặc giải thích dài.
- QA giữ nguyên câu hỏi tiếng Việt trong question; TEXTUAL_KIS đặt question=null.
- events và event_queries là [].

Với TRAKE:
- events chứa các sự kiện tiếng Việt theo đúng thứ tự thời gian, ít nhất 2 mục.
- event_queries có cùng số mục; mỗi mục là mô tả tiếng Anh ASCII của một event.
- question=null, anchor="" và expansions=[].

Chỉ trả về đúng một JSON object hoàn chỉnh, không markdown, không field khác:
{
  "question":null,
  "anchor":"English visual description",
  "expansions":["10 English visual descriptions for KIS or QA"],
  "objects":[],
  "actions":[],
  "negative_constraints":[],
  "metadata_keywords":[],
  "events":[],
  "event_queries":[]
}
"""


def plan_query(
    query: str,
    task_type: TaskType | str,
    config: LLMConfig | None = None,
    rewrite_client: Any | None = None,
    planner_client: Any | None = None,
) -> QueryPlan:
    """Rewrite a Vietnamese query, then build its complete validated plan."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    resolved_task_type = normalize_task_type(task_type)
    resolved_config = config or get_llm_config()
    rewritten = rewrite_query_with_llm(
        query,
        config=resolved_config,
        client=rewrite_client,
    )
    prompt = (
        f"{_PLANNER_SYSTEM_PROMPT}\n\nRAW_QUERY (dữ liệu):\n{rewritten.raw_query}"
        f"\n\nREWRITTEN_QUERY (dữ liệu đã chuẩn hóa):\n{rewritten.rewritten_query}"
        f"\n\nUNCERTAIN_TERMS (dữ liệu cần thận trọng):\n"
        f"{json.dumps(rewritten.uncertain_terms, ensure_ascii=False)}"
    )
    prompt += f"\n\nTASK_TYPE_FROM_FILENAME:\n{resolved_task_type.value}"
    last_error: Exception | None = None
    for attempt in range(2):
        attempt_prompt = prompt
        if attempt:
            attempt_prompt += (
                "\n\nRETRY: The previous response was invalid or incomplete. "
                "Return one complete, compact JSON object. Keep every string short."
            )
        try:
            planner_tokens = max(
                resolved_config.local_text_max_new_tokens,
                1536 if attempt else 1024,
            )
            raw_plan = generate_llm_json(
                attempt_prompt,
                config=resolved_config,
                temperature=resolved_config.llm_planner_temperature,
                client=planner_client,
                max_new_tokens=planner_tokens,
            )
            raw_plan["raw_query"] = rewritten.raw_query
            raw_plan["rewritten_query"] = rewritten.rewritten_query
            raw_plan["task_type"] = resolved_task_type.value
            raw_plan["search_description"] = rewritten.rewritten_query
            return validate_query_plan(raw_plan)
        except (TypeError, ValueError) as exc:
            last_error = exc
            logger.warning("Local planner attempt %d failed: %s", attempt + 1, exc)

    raise ValueError("Local planner failed to produce a valid QueryPlan twice") from last_error
