"""LLM query planning: Vietnamese semantic extraction plus CLIP queries."""

from __future__ import annotations

import json
from typing import Any

from .config import LLMConfig, get_llm_config
from .llm_client import generate_llm_json
from .rewrite import rewrite_query_with_llm
from .schemas import QueryPlan
from .task_types import TaskType, normalize_task_type
from .validator import validate_query_plan


_PLANNER_SYSTEM_PROMPT = r"""
Bạn là bộ lập QueryPlan cho hệ thống tìm kiếm video AIC. Nhận RAW_QUERY và
REWRITTEN_QUERY, rồi chỉ trả về một JSON QueryPlan hợp lệ.

NGÔN NGỮ VÀ BẢO TOÀN THÔNG TIN
- Mọi field semantic (trừ clip_queries và source) viết bằng tiếng Việt có dấu.
  clip_queries là mô tả hình ảnh tiếng Anh ASCII cho CLIP.
- Chỉ dùng thông tin có trong hai query. RAW_QUERY quyết định tên riêng, mã,
  chữ, quan hệ và thứ tự; REWRITTEN_QUERY chỉ sửa dấu/chính tả/viết tắt.
- Không bịa người, vật, thuộc tính, bối cảnh, nguồn camera hay đáp án. Điều
  không rõ dùng null, [] hoặc {}. negative_constraints chỉ chứa điều được nói rõ.
- Giữ quan hệ chủ thể - hành động - đối tượng trong entities.actions.
- source chỉ có thể là surveillance, sousveillance, dashcam, handheld,
  broadcast, mobile_phone hoặc unknown; không suy đoán source.

TASK_TYPE
- TASK_TYPE được cung cấp sẵn từ tên file của ban tổ chức. Không tự suy luận,
  không đổi task_type.
- TEXTUAL_KIS: tìm một cảnh/keyframe; question=null.
- QA: cần trả lời câu hỏi; question là nguyên câu hỏi tiếng Việt.

VISUAL_HINTS
- medium luôn là "video frame".
- capture_context=null nếu nguồn quay không được nêu. CCTV/camera giám sát cố
  định -> surveillance; camera đeo người -> sousveillance; camera hành trình
  -> dashcam; camera cầm tay -> handheld. Chỉ điền style/viewpoint/motion khi
  có bằng chứng.
- core_subjects lấy từ entities/objects; hypernyms, states, environment chỉ
  giữ thông tin an toàn và được nêu rõ.

CLIP_QUERIES
- Đây là danh sách cuối cùng, không phải pool candidate. Mỗi câu là mô tả
  trực quan độc lập có thể xuất hiện trong một frame; không phải câu hỏi/lệnh.
- TEXTUAL_KIS: 1 query chính, thêm tối đa 1 query khi có bằng chứng độc lập.
  QA: 1 query chính, thêm tối đa 3 query bằng chứng độc lập.
- Query mới phải thêm một bằng chứng rõ ràng (chủ thể/vật thể, hành động-quan
  hệ, bối cảnh, hoặc vùng logo/overlay/biển hiệu), không chỉ đổi từ đồng nghĩa.
- Không tạo query nhằm "nhìn kỹ" hay suy luận đáp án: cấm "showing clothing
  details", "of a specific color", "showing if", "to identify", "for OCR".
  Màu áo chưa biết, có áo khoác không, cách cầm đồ vật, hoặc nội dung chữ là
  việc VLM/OCR đọc SAU retrieval, không phải thuộc tính để bịa trong query.
- Chỉ gắn camera context khi được nêu rõ. Khi không biết source, dùng mô tả
  trung tính; không ép câu nào phải mở đầu bằng "a video frame showing".
- Với OCR/ASR chỉ mô tả vùng chữ/logo/đồng hồ/người nói đã được nêu; không
  đoán nội dung và không viết mục đích như "for OCR".

MINI FEW-SHOT (chỉ minh họa field khác biệt; JSON thật phải đủ schema)
KIS:
Input: Tìm cảnh xe buýt màu vàng dừng bên đường.
Output liên quan:
{"task_type":"TEXTUAL_KIS","question":null,"visual_hints":{"medium":"video frame","capture_context":null},"clip_queries":["a yellow bus stopped by the road"]}

QA:
Input: Tìm bốn phụ nữ cầm giấy trong chương trình truyền hình. Người thứ hai mặc áo màu gì?
Output liên quan:
{"task_type":"QA","question":"Người thứ hai mặc áo màu gì?","clip_queries":["four women standing side by side and holding sheets of paper in a television broadcast"]}
Không tạo query có màu áo giả định hoặc "showing clothing details".

CHỈ TRẢ VỀ đúng một JSON object, không array, không nhiều phương án, không
markdown/giải thích/field thừa:
{
  "task_type":"TEXTUAL_KIS | QA",
  "search_description":"mô tả tiếng Việt có dấu",
  "question":"câu hỏi tiếng Việt hoặc null",
  "entities":[{"name":"đối tượng","type":"person | object | vehicle | location | unknown","attributes":["thuộc tính"],"actions":[{"verb":"hành động","target":"đích hoặc null"}]}],
  "objects":["danh từ"], "actions":["hành động"], "scene":["bối cảnh"],
  "positive_constraints":["ràng buộc phải có"],
  "negative_constraints":["ràng buộc phải tránh"],
  "metadata_keywords":["từ khóa OCR/ASR"],
  "visual_hints":{"medium":"video frame","capture_context":{"source":"surveillance | sousveillance | dashcam | handheld | broadcast | mobile_phone | unknown","camera_style":"mô tả hoặc null","viewpoint":"mô tả hoặc null","camera_motion":"mô tả hoặc null","certainty":"explicit | metadata | unknown"},"core_subjects":["chủ thể"],"hypernyms":{"đối tượng":["siêu cấp an toàn"]},"states":["trạng thái"],"environment":["môi trường"]},
  "clip_queries":["concise English visual query"]
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
    prompt += """

OUTPUT CONTRACT OVERRIDE:
- Return an English ASCII `anchor` that is the closest literal visual translation.
- Return exactly 10 distinct English ASCII `expansions`.
 - Do not rely on `clip_queries`; downstream code derives three retrieval queries.
"""
    prompt += """

TRAKE CONTRACT:
- If TASK_TYPE_FROM_FILENAME is TRAKE, return at least two ordered `events`
  and exactly one English ASCII `event_queries` item per event.
- Each event query must describe only that event and preserve event order.
- For TEXTUAL_KIS and QA, return `events: []` and `event_queries: []`.
"""
    raw_plan = generate_llm_json(
        prompt,
        config=resolved_config,
        temperature=resolved_config.llm_planner_temperature,
        client=planner_client,
    )
    raw_plan["raw_query"] = rewritten.raw_query
    raw_plan["rewritten_query"] = rewritten.rewritten_query
    raw_plan["task_type"] = resolved_task_type.value
    if not raw_plan.get("anchor") and raw_plan.get("clip_queries"):
        raw_plan["anchor"] = raw_plan["clip_queries"][0]
    return validate_query_plan(raw_plan)
