# Hướng dẫn chạy hệ thống LLM

Tài liệu này hướng dẫn chạy độc lập luồng KIS, QA và TRAKE bằng bundle CLIP/FAISS local.

## 1. Cấu trúc thư mục

```text
LLM/
├── .env
├── run_inference.py
├── inputs/
├── outputs/
├── Keyframes/
└── video_search_bundle/clip-b32-btc-v1/
    ├── artifact_manifest.json
    ├── video.index
    ├── index_metadata.json
    └── lora_weights.pt
```

`video_search_bundle` là artifact dùng cho retrieval. Không cần source video hoặc mã preprocessing của project khác.

## 2. Cài đặt

Mở PowerShell tại thư mục LLM:

```powershell
cd "D:\AI\AIC 2026\LLM"
python -m pip install -r requirements.txt
```

Nếu dùng môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` dùng PyTorch có khả năng CUDA nhưng vẫn chạy được profile
CPU. Bitsandbytes chỉ được dùng khi profile GPU bật chế độ 4-bit.

Luôn chạy lệnh cài đặt bằng đúng Python/virtual environment dùng để chạy
`run_inference.py`.

Kiểm tra môi trường:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

Profile CPU chấp nhận `CUDA: False`. Profile GPU cần `CUDA: True` và phải hiện
đúng tên card NVIDIA.

## 3. Cấu hình `.env`

Tạo `.env` từ `.env.example`. Toàn bộ Rewrite, Planner, reranking và QA đều chạy local:

```env
LLM_PROVIDER=local
VLM_PROVIDER=local
LOCAL_MODEL=Qwen/Qwen3-VL-2B-Instruct
AIC_MODEL_CACHE_DIR=model_cache
LOCAL_DEVICE=cpu
LOCAL_LOAD_IN_4BIT=false
VLM_LOCAL_BATCH_SIZE=1
VLM_LOCAL_MAX_NEW_TOKENS=192
LLM_LOCAL_MAX_NEW_TOKENS=1536
VLM_LOCAL_MIN_PIXELS=200704
VLM_LOCAL_MAX_PIXELS=401408
LLM_REWRITE_TEMPERATURE=0
LLM_PLANNER_TEMPERATURE=0
VLM_CANDIDATE_LIMIT=10
VLM_WEIGHT=0.65

AIC_ARTIFACT_DIR=video_search_bundle/clip-b32-btc-v1
AIC_CLIP_MODEL_NAME=ViT-B/32
AIC_USE_LORA=true
AIC_DEVICE=cpu
AIC_KEYFRAMES_ROOT=Keyframes
```

Không cần API key. Không commit `.env` vào source code.

`AIC_MODEL_CACHE_DIR` điều khiển nơi lưu model Qwen và CLIP. Đường dẫn tương đối
được tính từ thư mục dự án; cấu hình trên lưu vào:

```text
LLM/model_cache/
├── huggingface/hub/   # Qwen3-VL
└── clip/              # OpenAI CLIP
```

Có thể đặt đường dẫn tuyệt đối sang ổ còn trống, ví dụ:

```env
AIC_MODEL_CACHE_DIR=E:/AIC-model-cache
```

Nếu model đang tải vào ổ C, nhấn `Ctrl+C`, đổi biến trên rồi chạy lại. Sau khi
model chạy thành công từ vị trí mới, cache cũ tại
`%USERPROFILE%\.cache\huggingface\hub` có thể được di chuyển hoặc xóa thủ công
để giải phóng ổ C.

Profile CPU dùng FP32, cần khoảng 10-12 GB RAM và sẽ chậm, phù hợp để kiểm tra
luồng. Trên máy NVIDIA 8 GB VRAM, đổi các dòng sau để chạy GPU:

```env
LOCAL_DEVICE=cuda
LOCAL_LOAD_IN_4BIT=true
VLM_LOCAL_BATCH_SIZE=8
VLM_CANDIDATE_LIMIT=40
AIC_DEVICE=cuda
```

## 4. Chuẩn bị query

Mỗi query BTC là một file `.txt` hoặc `.csv` riêng. Với chế độ `--input-dir`, file chỉ cần chứa nguyên văn câu truy vấn; loại tác vụ được lấy từ tên file:

```text
inputs/
├── query-1-kis.txt
├── query-2-qa.txt
└── query-4-trake.txt
```

Nội dung mỗi file chỉ chứa một câu truy vấn tiếng Việt.

Ví dụ `query-1-kis.txt`:

```text
Tìm một người đang mở laptop trong phòng.
```

Ví dụ `query-2-qa.txt`:

```text
Người phụ nữ đang mặc áo màu gì?
```

Ví dụ `query-4-trake.txt`:

```text
Một người bước vào cửa rồi ngồi xuống ghế.
```

Tên file phải kết thúc bằng một trong các hậu tố:

```text
-kis.txt
-qa.txt
-trake.txt
```

## 5. Chạy inference

Chạy KIS, QA và TRAKE:

```powershell
python run_inference.py `
  --input-dir inputs `
  --output-dir outputs/results `
  --checkpoint outputs/checkpoint.jsonl
```

Kết quả được ghi riêng cho từng query:

```text
outputs/results/
├── query-1-kis.csv
├── query-2-qa.csv
└── query-4-trake.csv
```

Các file theo query dùng định dạng nộp BTC và không có header:

```text
KIS:   video_id,frame_id
QA:    video_id,frame_id,answer
TRAKE: video_id,event_frame_1,event_frame_2,...
```

Checkpoint cho phép chạy tiếp các query chưa hoàn thành:

```text
outputs/checkpoint.jsonl
```

## 6. Bật VLM local

VLM dùng ảnh keyframe để rerank KIS hoặc trả lời QA. Với cấu hình mặc định,
Qwen3-VL chạy local bằng CPU và không gọi API. Chế độ này chậm; nên dùng để
smoke test ít query/candidate trước khi chuyển sang profile GPU:

```powershell
python -m pip install -r requirements.txt
```

Đặt ảnh theo cấu trúc:

```text
Keyframes/
├── L21_V001/
│   ├── 001.jpg
│   └── 002.jpg
└── L21_V002/
    └── 001.jpg
```

Tên thư mục video và tên ảnh phải khớp với `index_metadata.json`.

Chạy với VLM:

```powershell
python run_inference.py `
  --input-dir inputs `
  --output-dir outputs/results-vlm `
  --checkpoint outputs/checkpoint-vlm.jsonl `
  --with-vlm
```

Không dùng lại `checkpoint.jsonl` của bước 5. Checkpoint đánh dấu query đã hoàn thành;
nếu dùng lại file đó, bước 6 sẽ bỏ qua query và không gọi VLM. Kết quả VLM nằm trong
`outputs/results-vlm/`.

`VLM_LOCAL_BATCH_SIZE` giới hạn số ảnh local xử lý trong một lượt để tránh tràn RAM/VRAM.
`VLM_CANDIDATE_LIMIT` kiểm soát số keyframe được rerank; `VLM_WEIGHT` là trọng số
confidence của VLM khi trộn với CLIP/RRF.

TRAKE hiện dùng CLIP/FAISS và dynamic programming để giữ thứ tự frame; VLM không bắt buộc.

## 7. Chế độ CSV cũ

Có thể chạy một file CSV thay vì thư mục `.txt`:

```powershell
python run_inference.py `
  --queries inputs/queries.csv `
  --output outputs/submission.csv `
  --checkpoint outputs/checkpoint.jsonl
```

Ở chế độ `--queries`, CSV cần header:

```csv
query_id,query,task_type
KIS_001,Tìm một người đang mở laptop,TEXTUAL_KIS
QA_001,Người phụ nữ mặc áo màu gì?,QA
TRAKE_001,Người bước vào rồi ngồi xuống,TRAKE
```

## 8. Kiểm tra nhanh khi lỗi

- `Missing CLIP manifest`: sai `AIC_ARTIFACT_DIR`.
- Ổ C hết dung lượng khi tải model: đặt `AIC_MODEL_CACHE_DIR` sang ổ D/E rồi chạy lại.
- `LoRA manifest requires ...`: kiểm tra `AIC_USE_LORA=true` và `lora_weights.pt`.
- VLM không tìm thấy ảnh: kiểm tra `AIC_KEYFRAMES_ROOT` và đường dẫn keyframe.
- Lỗi thiết bị CPU: đặt `LOCAL_DEVICE=cpu`, `LOCAL_LOAD_IN_4BIT=false` và `AIC_DEVICE=cpu`.
- Lỗi thiết bị GPU: đặt hai device thành `cuda`, kiểm tra driver NVIDIA và `torch.cuda.is_available()` ở mục 2.
- `Local VLM response ... JSON object`: cập nhật branch `KIS_QA` mới nhất và đặt `LLM_LOCAL_MAX_NEW_TOKENS=1536`. Parser cứu các field hoàn chỉnh nếu output bị cắt; Rewrite có fallback và Planner tự thử lại một lần.
- Không có output: kiểm tra tên file có hậu tố `-kis`, `-qa` hoặc `-trake`.
