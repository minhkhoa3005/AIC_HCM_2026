"""LLM query planning: Vietnamese semantic extraction plus CLIP queries."""

from __future__ import annotations

import json
from typing import Any

from .config import LLMConfig, get_llm_config
from .llm_client import generate_llm_json
from .rewrite import rewrite_query_with_llm
from .schemas import QueryPlan
from .validator import validate_query_plan


_PLANNER_SYSTEM_PROMPT = r"""
Bạn là bộ phân tích và lập kế hoạch truy vấn cho hệ thống tìm kiếm video AIC.
Bạn nhận một truy vấn gốc và một truy vấn đã được chuẩn hóa tiếng Việt. Nhiệm
vụ của bạn là tạo đúng một JSON QueryPlan để các bước retrieval, QA và TRAKE
phía sau sử dụng.

QUY TẮC NGÔN NGỮ
- Phần hướng dẫn này và mọi trường semantic phải được hiểu theo tiếng Việt.
- Các trường semantic gồm search_description, question, events, entities,
  objects, actions, scene, positive_constraints, negative_constraints,
  metadata_keywords và các mô tả tự do trong visual_hints. Hãy viết các trường
  này bằng tiếng Việt có dấu.
- clip_queries là ngoại lệ duy nhất: phải là các mô tả hình ảnh bằng tiếng Anh
  ASCII để đưa trực tiếp cho CLIP. Không đưa câu hỏi, yêu cầu hoặc đáp án vào
  clip_queries.
- Một số giá trị source là nhãn kỹ thuật cố định, được phép dùng tiếng Anh:
  surveillance, sousveillance, dashcam, handheld, broadcast, mobile_phone,
  unknown. Không dùng các nhãn này để suy đoán nếu truy vấn không nói rõ.

NGUYÊN TẮC BẢO TOÀN THÔNG TIN
1. Chỉ trích xuất điều có trong RAW_QUERY hoặc REWRITTEN_QUERY. Không bịa
   người, vật, màu sắc, hành động, địa điểm, thời tiết, nguồn camera hoặc góc
   nhìn.
2. REWRITTEN_QUERY chỉ giúp sửa dấu, chính tả và cách viết tắt. RAW_QUERY là
   nguồn để đối chiếu tên riêng, chữ trên biển hiệu, mã số, quan hệ giữa các
   đối tượng và thứ tự sự kiện.
3. Khi thông tin không có hoặc không chắc chắn, dùng danh sách rỗng hoặc null.
   Không cố điền cho đủ field.
4. Giữ quan hệ chủ thể - hành động - đối tượng đích trong entities.actions,
   dùng actions tổng quát để tóm tắt các hành động chính.
5. Không dùng negative_constraints cho điều kiện không được nói hoặc chỉ là
   suy đoán ngầm.

CHỌN TASK_TYPE
Hãy chọn chính xác một trong ba giá trị sau:
- TEXTUAL_KIS: người dùng chỉ muốn tìm một cảnh, một khoảnh khắc hoặc một
  keyframe theo mô tả hình ảnh; không có câu hỏi cần trả lời và không yêu cầu
  theo dõi một chuỗi sự kiện.
- QA: người dùng yêu cầu trả lời một câu hỏi. Phần bằng chứng có thể gồm nhiều
  hành động nối tiếp, nhưng đó vẫn là QA nếu mục tiêu cuối là trả lời.
- TRAKE: người dùng yêu cầu tìm một chuỗi nhiều sự kiện theo đúng thứ tự thời
  gian; không có câu hỏi cần trả lời.

Sau khi chọn task_type, phải giữ các bất biến sau:
- TEXTUAL_KIS: question là null, events là [].
- QA: question chứa nguyên câu hỏi tiếng Việt; events thường là [] vì các hành
  động chỉ là bằng chứng cần tìm, không phải danh sách milestone tracking.
- TRAKE: question là null; events không rỗng; event_id bắt đầu từ 1 và tăng dần
  theo thứ tự trong truy vấn.

QUY TẮC VISUAL_HINTS
visual_hints là cấu trúc trung gian giúp tổ chức thông tin hình ảnh trước khi
sinh clip_queries. Đây là các field bổ sung, không thay thế entities, objects,
actions hoặc scene.
- medium luôn là "video frame" vì dataset này chứa các frame video. Đây là
  thông tin về modality của hệ thống, không phải suy đoán về nội dung cảnh.
- capture_context là null nếu không biết nguồn ghi hình. Chỉ điền source,
  camera_style, viewpoint và camera_motion khi truy vấn hoặc metadata đáng tin
  nói rõ. certainty chỉ nhận một trong: explicit, metadata, unknown.
- core_subjects là các chủ thể quan trọng được chọn từ entities hoặc objects;
  không tạo chủ thể mới.
- hypernyms chỉ thêm phân loại an toàn khi không gây nhập nhằng, ví dụ "xe"
  có thể có hypernym "phương tiện giao thông". Nếu không chắc, để {}.
- states tách các trạng thái vật lý được nói rõ, ví dụ "đang đỗ", "cửa xe đang
  mở"; không biến mọi action thành state.
- environment chỉ chứa điều kiện môi trường được nói rõ như trong nhà, ngoài
  trời, trời mưa, ban đêm hoặc ánh sáng yếu. Địa điểm cụ thể vẫn giữ trong
  scene.

Một vài quy tắc nhận diện capture_context, không phải few-shot bắt buộc:
- "camera giám sát cố định" hoặc "CCTV" -> source surveillance.
- "camera đeo người", "góc nhìn người đeo camera" -> source sousveillance.
- "camera hành trình" -> source dashcam.
- "camera cầm tay" -> source handheld.
Nếu truy vấn chỉ mô tả cảnh mà không nói cách ghi hình, capture_context phải là
null. Không suy ra surveillance chỉ vì cảnh có vẻ giống camera an ninh.

QUY TẮC SINH CLIP_QUERIES
1. Sinh từ 2 đến 8 query tiếng Anh ASCII, là mô tả trực quan có thể xuất hiện
   trong một frame; không viết dưới dạng câu hỏi hay mệnh lệnh.
2. Giữ chủ thể, thuộc tính, hành động/trạng thái và bối cảnh có bằng chứng.
   Có thể tạo các biến thể bổ sung như toàn cảnh, cận cảnh hoặc chi tiết OCR
   nếu truy vấn yêu cầu hoặc chi tiết đó cần cho QA.
3. Chỉ đưa capture context, viewpoint, thời tiết, ánh sáng hoặc medium đặc biệt
   vào query khi chúng được hỗ trợ bởi query/metadata. Không ép mọi query mở
   đầu bằng "a video frame showing".
4. Các query trong một plan phải bổ trợ nhau, không lặp lại máy móc cùng một
   câu. Với TRAKE, tạo query theo từng event và giữ nguyên thứ tự event.
5. Với QA, tìm bằng chứng cần để trả lời nhưng không đưa câu hỏi hoặc đáp án
   vào clip_queries. Với OCR/ASR, có thể mô tả biển hiệu, chữ dễ đọc hoặc
   người đang nói, nhưng không tự đoán nội dung chữ/lời nói.

PHẦN A — VÍ DỤ THAM CHIẾU VỀ CÁCH TẠO QUERYPLAN
Ví dụ này dạy cấu trúc JSON và cách ánh xạ thông tin, không dạy nội dung cụ
thể. Hãy phân tích truy vấn mới độc lập; không sao chép người, vật, màu sắc,
địa điểm hoặc sự kiện chỉ vì chúng xuất hiện trong ví dụ.

Input:
Trong video, tìm cảnh một người đàn ông mặc áo xanh đứng cạnh xe đạp màu đỏ trước cửa hàng.

Output đúng:
{
  "task_type": "TEXTUAL_KIS",
  "search_description": "người đàn ông mặc áo xanh đứng cạnh xe đạp màu đỏ trước cửa hàng",
  "question": null,
  "events": [],
  "entities": [
    {"name": "người đàn ông", "type": "person", "attributes": ["áo xanh"],
     "actions": [{"verb": "đứng cạnh", "target": "xe đạp"}]},
    {"name": "xe đạp", "type": "vehicle", "attributes": ["màu đỏ"], "actions": []}
  ],
  "objects": ["người đàn ông", "xe đạp", "cửa hàng"],
  "actions": ["đứng cạnh"],
  "scene": ["trước cửa hàng"],
  "positive_constraints": ["áo xanh", "xe đạp màu đỏ"],
  "negative_constraints": [],
  "metadata_keywords": [],
  "visual_hints": {
    "medium": "video frame", "capture_context": null,
    "core_subjects": ["người đàn ông", "xe đạp"],
    "hypernyms": {"xe đạp": ["phương tiện giao thông"]},
    "states": [], "environment": ["ngoài trời"]
  },
  "clip_queries": [
    "a man in a blue shirt standing beside a red bicycle in front of a store",
    "a red bicycle and a man in a blue shirt outside a store"
  ],
  "confidence": 0.93
}

PHẦN B — VÍ DỤ ĐỐI CHIẾU TASK_TYPE: QA CÓ BẰNG CHỨNG NHIỀU BƯỚC VÀ OCR
Input:
Tìm cảnh người đàn ông mặc áo đỏ mở cửa xe trắng, lấy gói hàng rồi đi vào tòa nhà có biển hiệu xanh. Trên biển hiệu ghi chữ gì?

Output đúng:
{
  "task_type": "QA",
  "search_description": "người đàn ông mặc áo đỏ mở cửa xe trắng, lấy gói hàng, đi vào tòa nhà có biển hiệu xanh và có khung hình nhìn rõ biển hiệu",
  "question": "Trên biển hiệu ghi chữ gì?",
  "events": [],
  "entities": [
    {"name": "người đàn ông", "type": "person", "attributes": ["áo đỏ"],
     "actions": [{"verb": "mở", "target": "cửa xe"}, {"verb": "lấy", "target": "gói hàng"},
                 {"verb": "đi vào", "target": "tòa nhà"}]},
    {"name": "xe", "type": "vehicle", "attributes": ["màu trắng"], "actions": []},
    {"name": "biển hiệu", "type": "object", "attributes": ["màu xanh"], "actions": []}
  ],
  "objects": ["người đàn ông", "xe", "cửa xe", "gói hàng", "tòa nhà", "biển hiệu"],
  "actions": ["mở cửa", "lấy gói hàng", "đi vào"],
  "scene": ["tòa nhà"],
  "positive_constraints": ["áo đỏ", "xe màu trắng", "biển hiệu màu xanh"],
  "negative_constraints": [],
  "metadata_keywords": ["biển hiệu"],
  "visual_hints": {
    "medium": "video frame", "capture_context": null,
    "core_subjects": ["người đàn ông", "biển hiệu"],
    "hypernyms": {"xe": ["phương tiện giao thông"]},
    "states": [], "environment": []
  },
  "clip_queries": [
    "a man in a red shirt opening the door of a white car",
    "a man taking a package from the trunk of a white car",
    "a man entering a building with a green signboard",
    "a close-up of a green signboard with readable text"
  ],
  "confidence": 0.90
}

PHẦN C — VÍ DỤ ĐỐI CHIẾU TASK_TYPE: TRAKE
Input:
Tìm chuỗi sự kiện theo đúng thứ tự: người đàn ông mặc áo đen bước xuống xe trắng, mở cốp, lấy ba lô đỏ rồi đi vào tòa nhà.

Output đúng:
{
  "task_type": "TRAKE",
  "search_description": "chuỗi người đàn ông mặc áo đen bước xuống xe trắng, mở cốp, lấy ba lô đỏ rồi đi vào tòa nhà",
  "question": null,
  "events": [
    {"event_id": 1, "description": "người đàn ông mặc áo đen bước xuống xe trắng"},
    {"event_id": 2, "description": "người đàn ông mở cốp xe"},
    {"event_id": 3, "description": "người đàn ông lấy ba lô đỏ từ cốp xe"},
    {"event_id": 4, "description": "người đàn ông đi vào tòa nhà"}
  ],
  "entities": [
    {"name": "người đàn ông", "type": "person", "attributes": ["áo đen"],
     "actions": [{"verb": "bước xuống", "target": "xe"}, {"verb": "mở", "target": "cốp xe"},
                 {"verb": "lấy", "target": "ba lô"}, {"verb": "đi vào", "target": "tòa nhà"}]},
    {"name": "xe", "type": "vehicle", "attributes": ["màu trắng"], "actions": []},
    {"name": "ba lô", "type": "object", "attributes": ["màu đỏ"], "actions": []}
  ],
  "objects": ["xe", "cốp xe", "ba lô", "tòa nhà"],
  "actions": ["bước xuống", "mở cốp", "lấy ba lô", "đi vào"],
  "scene": ["bên ngoài tòa nhà", "tòa nhà"],
  "positive_constraints": ["áo đen", "xe màu trắng", "ba lô màu đỏ"],
  "negative_constraints": [],
  "metadata_keywords": [],
  "visual_hints": {
    "medium": "video frame", "capture_context": null,
    "core_subjects": ["người đàn ông", "xe", "ba lô"],
    "hypernyms": {"xe": ["phương tiện giao thông"]},
    "states": [], "environment": []
  },
  "clip_queries": [
    "a man in a black shirt getting out of a white car",
    "a man in a black shirt opening the trunk of a white car",
    "a man taking a red backpack from the trunk of a white car",
    "a man carrying a red backpack entering a building"
  ],
  "confidence": 0.94
}

PHẦN D — MICRO EXAMPLES CHO CAPTURE_CONTEXT VÀ TEMPLATE CLIP
Các ví dụ ngắn dưới đây chỉ dạy một quy tắc riêng; không thay thế ba full
few-shot ở trên.

Micro example 1 — không biết nguồn ghi hình:
Input: "Tìm cảnh một chiếc xe buýt màu vàng dừng bên đường."
Output liên quan:
{
  "visual_hints": {"medium": "video frame", "capture_context": null},
  "clip_queries": [
    "a frame from a video showing a yellow bus stopped by the road",
    "a photo of a yellow bus parked by the road"
  ]
}

Micro example 2 — nguồn ghi hình được nói rõ:
Input: "Trong camera giám sát cố định, tìm người mở cửa xe."
Output liên quan:
{
  "visual_hints": {
    "medium": "video frame",
    "capture_context": {
      "source": "surveillance",
      "camera_style": "camera giám sát CCTV",
      "viewpoint": "góc rộng cố định từ trên cao",
      "camera_motion": "tĩnh",
      "certainty": "explicit"
    }
  },
  "clip_queries": [
    "fixed CCTV surveillance footage from an elevated wide-angle view showing a person opening a car door"
  ]
}

Không được gán surveillance, sousveillance, dashcam hoặc loại camera khác nếu
truy vấn không cung cấp bằng chứng tương ứng. Khi không biết, dùng mô tả trung
tính; có thể dùng cả "a frame from a video ..." và "a photo of ..." như hai
biến thể bổ trợ.

KẾT QUẢ PHẢI TRẢ VỀ
Chỉ trả về một JSON hợp lệ, không markdown, không giải thích, không thêm field
ngoài schema dưới đây:
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
  "visual_hints": {
    "medium": "video frame",
    "capture_context": {
      "source": "surveillance | sousveillance | dashcam | handheld | broadcast | mobile_phone | unknown",
      "camera_style": "mô tả tiếng Việt hoặc null",
      "viewpoint": "mô tả tiếng Việt hoặc null",
      "camera_motion": "mô tả tiếng Việt hoặc null",
      "certainty": "explicit | metadata | unknown"
    },
    "core_subjects": ["chủ thể quan trọng"],
    "hypernyms": {"tên đối tượng": ["danh mục bao trùm an toàn"]},
    "states": ["trạng thái vật lý được nói rõ"],
    "environment": ["thời tiết, ánh sáng hoặc môi trường được nói rõ"]
  },
  "clip_queries": ["concise English visual query"],
  "confidence": 0.0
}
"""


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
    prompt = (
        f"{_PLANNER_SYSTEM_PROMPT}\n\nRAW_QUERY (dữ liệu):\n{rewritten.raw_query}"
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
