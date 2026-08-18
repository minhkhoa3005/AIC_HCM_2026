# AIC 2026 Video Query

MVP truy vấn video cho AI Challenge 2026. Hệ thống chuẩn hóa câu hỏi tiếng Việt, lập kế hoạch truy vấn bằng LLM, rồi chuyển thành mô tả hình ảnh tiếng Anh cho CLIP.

## Luồng xử lý

```text
Vietnamese query
  -> task_type from organizer filename/query_id
  -> rewrite (sửa dấu, chính tả, viết tắt)
  -> planner (QueryPlan đã kiểm tra)
  -> English ASCII clip_queries
  -> retrieval
  -> rerank / QA
  -> Prediction / CSV
```

Có đúng 2 lần gọi LLM: `rewrite_query_with_llm()` và `plan_query()`. `task_type` được lấy từ tên file/query_id của ban tổ chức hoặc truyền trực tiếp vào `run_query()`, không để LLM nhận dạng. `QueryPlan` giữ cả `raw_query` và `rewritten_query`; các trường ngữ nghĩa dùng tiếng Việt có dấu, chỉ `clip_queries` là tiếng Anh ASCII.

## Các gói mã nguồn

```text
llm/          Rewrite, lập QueryPlan, validate, cấu hình Gemini và xoay API key
retrieval/    Kiểu dữ liệu retrieval, interface CLIP và mock dùng offline
pipeline/     Điều phối đơn/lô truy vấn, RRF rerank, QA, checkpoint JSONL
submission/   Prediction, override thủ công và xuất CSV có cấu hình
```

- `TEXTUAL_KIS`: trả về các keyframe được xếp hạng.
- `QA`: chọn bằng chứng tốt nhất; câu trả lời dự phòng lấy OCR, rồi ASR.

`retrieval.search_clip_text()` vẫn là interface để inject backend bên ngoài. Ngoài ra,
`retrieval.LocalBundleRetriever` có thể đọc trực tiếp bundle `clip-b32-btc-v1`
và nhận một CLIP text encoder từ bên ngoài. Encoder không bị tạo lại trong LLM,
nhằm bảo đảm dùng đúng CLIP/LoRA tương thích với image index. Adapter hỗ trợ
`search_many()` và kiểm tra manifest, dimension, metric, vector_count và thứ tự
metadata trước khi đưa kết quả vào Weighted RRF.

## Cấu hình và tài liệu kèm theo

Sao chép `.env.example` thành `.env` và đặt khóa theo biến duy nhất `LLM_API_KEYS` (phân cách bằng dấu phẩy). Provider hiện tại là Gemini (`google-genai`). `requirements.txt` chứa các phụ thuộc Python.

`testlist.txt` là bộ truy vấn mẫu QA và TEXTUAL_KIS; `test.txt` là lệnh chạy thử thủ công. Các PDF, ảnh và ghi chú ở thư mục gốc là tài liệu/tham chiếu dự án, không phải mã nguồn chạy.

## Cài đặt

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Khi tích hợp retrieval thật, giữ nguyên contract
`list[str] -> list[QueryRetrievalResult]` để pipeline và export CSV hoạt động
không đổi. `pipeline.run_query()` sẽ kiểm tra query index, query text, số lượng
candidate, frame_id và pts_time trước khi fusion.
