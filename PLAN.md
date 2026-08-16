# Kế Hoạch Hoàn Thiện Cho Vòng Sơ Tuyển AIC 2026

## Mục Tiêu

Xây dựng hệ thống tìm kiếm khoảnh khắc video có thể nộp đáp án theo ba dạng truy vấn của đề: Textual KIS, Q&A và TRAKE. Trọng tâm là xếp hạng đúng trong Top 1/5/20/50/100 và xuất `frame_id` chính xác.

## Đã Hoàn Thành

- [x] Pipeline giải nén và cấu trúc dữ liệu BTC (`extract_btc_data.py`).
- [x] Import metadata tổng hợp (`metadata.jsonl`) tích hợp FPS, timestamp, object tags.
- [x] Trích xuất lời thoại âm thanh bằng Faster-Whisper (`transcribe.py`).
- [x] Fine-tune CLIP backbone bằng **LoRA (Low-Rank Adaptation)** chống Catastrophic Forgetting (`lora.py`, `train_lora_clip.py`).
- [x] Cấu hình linh hoạt GPU/CPU qua cờ `--device` và file `.env`.
- [x] Đẩy vector và metadata vào FAISS Local Index.
- [x] Trích xuất hình ảnh keyframe kết quả tìm kiếm kèm xuất báo cáo Markdown trực quan.

## Kế Hoạch Tiếp Theo

| Ưu tiên | Việc làm | Kết quả mong muốn |
|---|---|---|
| P0 | Chạy lại preprocessing với video thật để ghi `fps`, `start_frame`, `end_frame`, `frame_id` | Frame nộp không bị lệch do giả định FPS |
| P0 | Tạo bộ query kiểm thử cho KIS/Q&A/TRAKE | Đo được R@1/5/20/50/100 nội bộ |
| P1 | Huấn luyện LoRA trên toàn bộ dataset L21/L22 | Tăng độ chính xác zero-shot retrieval trên dữ liệu tiếng Việt/AIC |
| P1 | Thêm rerank trong cùng video cho TRAKE | Frame theo chuỗi sự kiện ổn định hơn |
| P2 | Tích hợp VLM/OCR trả lời câu hỏi Q&A tự động | Giảm thao tác trả lời thủ công |

## Kiểm Thử & Đánh Giá

- **KIS**: Kiểm tra `video_id` đúng và `frame_id` nằm trong đoạn đáp án.
- **Q&A**: Kiểm tra `video_id`, `frame_id` và answer khớp ngữ nghĩa.
- **TRAKE**: Kiểm tra đúng video trước, sau đó tính tỉ lệ frame đúng trên từng khoảnh khắc.
- Báo cáo điểm bằng trung bình `R@1`, `R@5`, `R@20`, `R@50`, `R@100`.
