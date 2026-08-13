# AIC 2026 Video Query

Pipeline truy vấn video cho AI Challenge 2026. Hệ thống dùng LLM cho hai bước
hiểu ngôn ngữ, sau đó chuyển kết quả thành danh sách truy vấn tiếng Anh cho
CLIP.

## Luồng Chính

```text
raw_query tiếng Việt
-> rewrite_query_with_llm
   -> rewritten_query tiếng Việt có dấu
-> plan_query
   -> task_type
   -> QueryPlan semantic fields tiếng Việt có dấu
   -> clip_queries tiếng Anh ASCII
-> build_clip_queries
-> retrieval.search_clip_text
-> rerank / QA / TRAKE
-> submission
```

Có đúng hai lần gọi LLM:

1. Rewrite: chỉ sửa chính tả, thiếu dấu và viết tắt; không dịch hoặc suy diễn.
2. Planner: xác định loại truy vấn, bóc tách `QueryPlan` và sinh `clip_queries`.

Query gốc luôn được giữ trong `QueryPlan.raw_query`. Bản rewrite nằm trong
`QueryPlan.rewritten_query` để có thể kiểm tra semantic drift.

## Các Loại Truy Vấn

- `TEXTUAL_KIS`: tìm một khoảnh khắc hoặc keyframe theo mô tả hình ảnh.
- `QA`: tìm bằng chứng hình ảnh rồi trả lời câu hỏi.
- `TRAKE`: tìm chuỗi sự kiện theo đúng thứ tự thời gian.

## Contract

```python
QueryPlan(
    raw_query="Tìm cảnh ng đàn ông mặc áo đỏ mở cửa xe trắng.",
    rewritten_query="Tìm cảnh người đàn ông mặc áo đỏ mở cửa xe trắng.",
    task_type=TaskType.TEXTUAL_KIS,
    search_description="người đàn ông mặc áo đỏ mở cửa xe trắng",
    objects=["người đàn ông", "xe trắng"],
    actions=["mở cửa"],
    clip_queries=[
        "a man in a red shirt opening the door of a white car",
        "a white car with an open door",
    ],
)
```

Các trường semantic giữ tiếng Việt có dấu. Chỉ `clip_queries` dùng tiếng Anh
ASCII vì đây là input của CLIP. `build_clip_queries` không dịch hoặc tự tạo
fallback; nó chỉ kiểm tra và trả về output của planner.

## Cấu Trúc

```text
llm/
  rewrite.py       Rewrite tiếng Việt có dấu
  planner.py       Planner đầy đủ và prompt inline
  schemas.py       QueryPlan, Candidate, Entity, TrakeEvent
  validator.py     Validate JSON thành QueryPlan
  query_builder.py Adapter QueryPlan -> list[str] cho CLIP
  llm_client.py    Adapter provider và JSON response
  providers/       API key pool dùng cho xoay key

retrieval/         Contract và mock CLIP retrieval
pipeline/          Controller, rerank, QA, TRAKE, checkpoint
submission/        Xuất CSV và cấu hình submission
```

## Cấu Hình

```env
LLM_PROVIDER=gemini
LLM_API_KEYS=key1,key2,key3
LLM_MODEL=gemini-2.5-flash
LLM_REWRITE_TEMPERATURE=0
LLM_PLANNER_TEMPERATURE=0
```

## Chạy Test

```powershell
pytest -q
```
