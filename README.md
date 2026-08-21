# AIC 2026 Video Query

MVP truy vấn video cho AI Challenge 2026. Hệ thống chuẩn hóa câu hỏi tiếng Việt, lập kế hoạch truy vấn bằng LLM, rồi chuyển thành mô tả hình ảnh tiếng Anh cho CLIP.

## Luồng xử lý

```text
Vietnamese query
  -> task_type from organizer filename/query_id
  -> rewrite (sửa dấu, chính tả, viết tắt)
  -> planner (QueryPlan đã kiểm tra)
  -> English anchor + expansions / ordered event queries
  -> retrieval
  -> rerank / QA
  -> Prediction / CSV
```

Rewrite và Planner dùng chung model local. Rewrite giữ nguyên raw query nếu model
trả JSON lỗi; Planner tự thử lại một lần với contract rút gọn. `task_type` được
lấy từ tên file/query_id của ban tổ chức hoặc truyền trực tiếp vào `run_query()`,
không để model nhận dạng. `QueryPlan` giữ cả `raw_query` và `rewritten_query`;
anchor, expansions và event_queries dùng tiếng Anh ASCII cho CLIP.

## Các gói mã nguồn

```text
llm/          Rewrite, lập QueryPlan, validate và sinh JSON bằng model local
retrieval/    Kiểu dữ liệu retrieval và kết nối CLIP/FAISS local
pipeline/     Điều phối đơn/lô truy vấn, RRF rerank, QA, checkpoint JSONL
submission/   Kiểm tra prediction và xuất CSV đúng định dạng BTC
```

- `TEXTUAL_KIS`: trả về các keyframe được xếp hạng.
- `QA`: chọn bằng chứng tốt nhất; câu trả lời dự phòng lấy OCR, rồi ASR.
- `TRAKE`: tìm từng event và ghép chuỗi frame tăng dần theo thời gian.

`retrieval.search_clip_text()` vẫn là interface để inject backend bên ngoài. Ngoài ra,
`retrieval.LocalBundleRetriever` có thể đọc trực tiếp bundle `clip-b32-btc-v1`
và nhận một CLIP text encoder từ bên ngoài. Encoder không bị tạo lại trong LLM,
nhằm bảo đảm dùng đúng CLIP/LoRA tương thích với image index. Adapter hỗ trợ
`search_many()` và kiểm tra manifest, dimension, metric, vector_count và thứ tự
metadata trước khi đưa kết quả vào Weighted RRF.

## Cấu hình và tài liệu kèm theo

Sao chép `.env.example` thành `.env`. Toàn bộ Rewrite, Planner, reranking và QA
chạy bằng model local trên CUDA; không cần API key. `requirements.txt` là file
dependency duy nhất và bao gồm CUDA PyTorch, CLIP, Transformers, Accelerate và
bitsandbytes.

## Cài đặt

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Khi tích hợp retrieval thật, giữ nguyên contract
`list[str] -> list[QueryRetrievalResult]` để pipeline và export CSV hoạt động
không đổi. `pipeline.run_query()` sẽ kiểm tra query index, query text, số lượng
candidate, frame_id và pts_time trước khi fusion.
