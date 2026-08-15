# AIC 2026 Video Query

MVP truy vấn video cho AI Challenge 2026. Hệ thống chuẩn hóa câu hỏi tiếng Việt, lập kế hoạch truy vấn bằng LLM, rồi chuyển thành mô tả hình ảnh tiếng Anh cho CLIP.

## Luồng xử lý

```text
Vietnamese query
  -> rewrite (sửa dấu, chính tả, viết tắt)
  -> planner (QueryPlan đã kiểm tra)
  -> English ASCII clip_queries
  -> retrieval
  -> rerank / QA / TRAKE
  -> Prediction / CSV
```

Có đúng 2 lần gọi LLM: `rewrite_query_with_llm()` và `plan_query()`. `QueryPlan` giữ cả `raw_query` và `rewritten_query`; các trường ngữ nghĩa dùng tiếng Việt có dấu, chỉ `clip_queries` là tiếng Anh ASCII.

## Các gói mã nguồn

```text
llm/          Rewrite, lập QueryPlan, validate, cấu hình Gemini và xoay API key
retrieval/    Kiểu dữ liệu retrieval, interface CLIP và mock dùng offline
pipeline/     Điều phối đơn/lô truy vấn, RRF rerank, QA, TRAKE, checkpoint JSONL
submission/   Prediction, override thủ công và xuất CSV có cấu hình
```

- `TEXTUAL_KIS`: trả về các keyframe được xếp hạng.
- `QA`: chọn bằng chứng tốt nhất; câu trả lời dự phòng lấy OCR, rồi ASR.
- `TRAKE`: ghép một chuỗi frame tăng dần theo thứ tự sự kiện.

`retrieval.search_clip_text()` hiện là interface chưa cài CLIP/FAISS. Dùng `retrieval.mock_search_clip_text` để kiểm tra luồng offline hoặc truyền hàm retrieval thật vào `pipeline.run_query()` / `pipeline.run_batch()`.

## Cấu hình và tài liệu kèm theo

Sao chép `.env.example` thành `.env` và đặt khóa theo biến duy nhất `LLM_API_KEYS` (phân cách bằng dấu phẩy). Provider hiện tại là Gemini (`google-genai`). `requirements.txt` chứa các phụ thuộc Python.

`testlist.txt` là bộ truy vấn mẫu QA, TEXTUAL_KIS và TRAKE; `test.txt` là lệnh chạy thử thủ công. Các PDF, ảnh và ghi chú ở thư mục gốc là tài liệu/tham chiếu dự án, không phải mã nguồn chạy.

## Cài đặt

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chưa có CLI và chưa có thư mục kiểm thử tự động trong checkout hiện tại. Khi tích hợp retrieval thật, giữ nguyên contract `list[str] -> list[QueryRetrievalResult]` để pipeline và export CSV hoạt động không đổi.
