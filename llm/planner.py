"""Two-stage LLM query planning: Vietnamese rewrite, then QueryPlan output."""

from __future__ import annotations

import json
from typing import Any

from .config import LLMConfig, get_llm_config
from .llm_client import generate_llm_json
from .rewrite import rewrite_query_with_llm
from .schemas import QueryPlan
from .validator import validate_query_plan


def plan_query(
    query: str,
    config: LLMConfig | None = None,
    rewrite_client: Any | None = None,
    planner_client: Any | None = None,
) -> QueryPlan:
    """Rewrite a Vietnamese query, then build its complete validated plan."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    resolved_config = config or get_llm_config()
    rewritten = rewrite_query_with_llm(
        query,
        config=resolved_config,
        client=rewrite_client,
    )

    # Keep the complete planner contract beside the call so the prompt and
    # output schema remain visible together.
    system_prompt = """
Bạn là LLM lập kế hoạch truy vấn cho hệ thống tìm kiếm video AIC.
Bạn nhận một truy vấn gốc và một bản viết lại tiếng Việt có dấu. Hãy dùng
bản viết lại để hiểu câu, nhưng đối chiếu truy vấn gốc để không làm mất tên riêng,
chữ trên biển hiệu, mã số, quan hệ giữa các đối tượng và thứ tự sự kiện.

Xác định task_type:
- TEXTUAL_KIS: tìm một khoảnh khắc hoặc keyframe theo mô tả hình ảnh.
- QA: tìm khoảnh khắc làm bằng chứng, sau đó trả lời câu hỏi.
- TRAKE: tìm một chuỗi nhiều sự kiện theo đúng thứ tự thời gian.

Quy tắc semantic fields:
1. Tất cả search_description, question, events, entities, objects, actions, scene,
   positive_constraints, negative_constraints và metadata_keywords phải giữ tiếng Việt có dấu.
2. Không dùng tiếng Việt không dấu trong các trường ngữ nghĩa.
3. Không bịa đặt chi tiết không có trong truy vấn. Khi không chắc, giữ cách diễn đạt trung lập.
4. QA: đặt nguyên câu hỏi vào question; search_description chỉ mô tả bằng chứng cần tìm.
5. TRAKE: events phải có event_id bắt đầu từ 1 và tăng dần theo thứ tự trong truy vấn.
6. entities phải giữ quan hệ ai làm gì với đối tượng nào qua actions.target.
7. negative_constraints chỉ chứa điều kiện phủ định thực sự có trong truy vấn.

Quy tắc clip_queries:
1. Sinh 2–8 chuỗi mô tả hình ảnh bằng tiếng Anh cho CLIP.
2. Chỉ clip_queries được viết bằng tiếng Anh ASCII; không chứa tiếng Việt.
3. Mỗi chuỗi phải cụ thể, giữ các đối tượng, hành động, màu sắc, bối cảnh và quan hệ quan trọng.
4. TEXTUAL_KIS: tạo các biến thể bổ trợ cho cùng khoảnh khắc.
5. QA: tập trung vào bằng chứng hình ảnh, OCR, ASR hoặc chi tiết cần trả lời; không đưa câu hỏi tiếng Việt vào clip_queries.
6. TRAKE: tạo query theo từng event và giữ đúng thứ tự.
7. Không tự thêm chi tiết hình ảnh chưa được nêu.

Ví dụ few-shot bắt buộc tham chiếu:
Input:
Trong video, tìm cảnh người phụ nữ mặc áo vàng chạy từ bên trái đến chiếc xe máy màu đỏ,
dừng xe, mở cốp lấy một túi giấy màu trắng rồi bước vào cửa hàng có biển hiệu màu xanh.
Hãy cho biết trên biển hiệu có chữ gì và người phụ nữ cầm túi bằng tay nào.

Output đúng:
{
  "task_type": "QA",
  "search_description": "cảnh người phụ nữ mặc áo vàng chạy đến xe máy màu đỏ, dừng xe, mở cốp lấy túi giấy màu trắng rồi bước vào cửa hàng có biển hiệu màu xanh",
  "question": "Trên biển hiệu có chữ gì và người phụ nữ cầm túi bằng tay nào?",
  "events": [],
  "entities": [],
  "objects": ["người phụ nữ", "xe máy", "cốp xe", "túi giấy", "cửa hàng", "biển hiệu"],
  "actions": ["chạy", "dừng xe", "mở cốp", "lấy túi giấy", "bước vào"],
  "scene": ["cửa hàng"],
  "positive_constraints": ["áo vàng", "xe máy màu đỏ", "túi giấy màu trắng", "biển hiệu màu xanh"],
  "negative_constraints": [],
  "metadata_keywords": ["biển hiệu"],
  "clip_queries": [
    "a woman in a yellow shirt running from the left toward a red motorcycle",
    "a woman in a yellow shirt stopping at a red motorcycle and opening its storage compartment",
    "a woman in a yellow shirt taking a white paper bag from a red motorcycle",
    "a woman in a yellow shirt holding a white paper bag and entering a store with a blue sign",
    "a close-up of a blue store sign",
    "a close-up of a woman holding a white paper bag with her hands visible"
  ],
  "confidence": 0.95
}

Khi tạo clip_queries, hãy bắt chước dạng mô tả hình ảnh của ví dụ trên.
Không sao chép các sự kiện hoặc đối tượng của ví dụ vào truy vấn mới.
Không viết clip_queries dưới dạng câu hỏi, mệnh lệnh hoặc yêu cầu suy luận.

Chỉ trả về một JSON hợp lệ theo schema sau, không markdown:
{
  "task_type": "TEXTUAL_KIS | QA | TRAKE",
  "search_description": "mô tả bằng chứng hình ảnh bằng tiếng Việt có dấu",
  "question": "câu hỏi tiếng Việt có dấu hoặc null",
  "events": [{"event_id": 1, "description": "sự kiện tiếng Việt có dấu"}],
  "entities": [{
    "name": "tên đối tượng",
    "type": "person | object | vehicle | location | unknown",
    "attributes": ["thuộc tính tiếng Việt có dấu"],
    "actions": [{"verb": "hành động", "target": "đối tượng đích hoặc null"}]
  }],
  "objects": ["danh từ tiếng Việt có dấu"],
  "actions": ["hành động tiếng Việt có dấu"],
  "scene": ["bối cảnh tiếng Việt có dấu"],
  "positive_constraints": ["ràng buộc phải có"],
  "negative_constraints": ["ràng buộc phải tránh"],
  "metadata_keywords": ["từ khóa OCR/ASR/biển hiệu"],
  "clip_queries": ["concise English visual query"],
  "confidence": 0.0
}
"""
    prompt = (
        f"{system_prompt}\n\nRAW_QUERY (dữ liệu):\n{rewritten.raw_query}"
        f"\n\nREWRITTEN_QUERY (dữ liệu đã chuẩn hóa):\n{rewritten.rewritten_query}"
        f"\n\nUNCERTAIN_TERMS (dữ liệu cần thận trọng):\n"
        f"{json.dumps(rewritten.uncertain_terms, ensure_ascii=False)}"
    )
    raw_plan = generate_llm_json(
        prompt,
        config=resolved_config,
        temperature=resolved_config.llm_planner_temperature,
        client=planner_client,
    )
    raw_plan["raw_query"] = rewritten.raw_query
    raw_plan["rewritten_query"] = rewritten.rewritten_query
    return validate_query_plan(raw_plan)
