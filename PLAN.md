# Kế Hoạch Hoàn Thiện Cho Vòng Sơ Tuyển AIC 2026

## Mục Tiêu

Xây dựng hệ thống tìm kiếm khoảnh khắc video có thể nộp đáp án theo ba dạng truy vấn của đề: Textual KIS, Q&A và TRAKE. Trọng tâm là xếp hạng đúng trong Top 1/5/20/50/100 và xuất `frame_id` chính xác.

## Việc Đã Có

- Pipeline cắt video thành clip và cache metadata theo video.
- Transcript bằng faster-whisper.
- Embedding lai transcript + keyframe bằng CLIP.
- FAISS index và API `/search`.
- Frontend xem lại video tại timestamp và copy dòng submission.

## Việc Nên Làm Tiếp

| Ưu tiên | Việc làm | Kết quả mong muốn |
|---|---|---|
| P0 | Chạy lại preprocessing với video thật để ghi `fps`, `start_frame`, `end_frame`, `frame_id` | Frame nộp không bị lệch do giả định FPS |
| P0 | Tạo bộ query kiểm thử cho KIS/Q&A/TRAKE | Đo được R@1/5/20/50/100 nội bộ |
| P1 | Import keyframes và CLIP features do BTC cấp nếu có | Không phải encode lại toàn bộ video, tăng tốc build index |
| P1 | Kết hợp object JSON vào `search_text` | Truy vấn vật thể/màu sắc/hành động chính xác hơn |
| P1 | Thêm rerank trong cùng video cho TRAKE | Frame theo chuỗi sự kiện ổn định hơn |
| P2 | Thêm module hỗ trợ answer Q&A bằng VLM/OCR nếu có GPU/API | Giảm thao tác trả lời thủ công |

## Kiểm Thử

- Với KIS: kiểm tra `video_id` đúng và `frame_id` nằm trong đoạn đáp án.
- Với Q&A: kiểm tra `video_id`, `frame_id` và answer khớp ngữ nghĩa.
- Với TRAKE: kiểm tra đúng video trước, sau đó tính tỉ lệ frame đúng trên từng khoảnh khắc.
- Báo cáo điểm bằng trung bình `R@1`, `R@5`, `R@20`, `R@50`, `R@100`.

## Rủi Ro

- Metadata cũ không có FPS sẽ làm frame chỉ là ước lượng 25 FPS.
- Clip quá dài dễ trả frame trung tâm chưa trúng semantic keyframe ngắn dưới 10 frame.
- TRAKE cần căn chỉnh nhiều khoảnh khắc, vì vậy top result chỉ nên là đề xuất ban đầu để người dùng xem lại và chỉnh.
