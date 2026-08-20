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

Nếu dùng GPU NVIDIA, cài bộ phụ thuộc CUDA:

```powershell
python -m pip install -r requirements-cuda.txt
```

Kiểm tra CUDA:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

Kết quả cần có `CUDA: True`. Nếu máy không có GPU NVIDIA hoặc PyTorch vẫn là
bản CPU, đặt `AIC_DEVICE=cpu` trong `.env` thay vì `cuda`.

## 3. Cấu hình `.env`

Tạo `.env` từ `.env.example`, sau đó điền API key Gemini:

```env
LLM_PROVIDER=gemini
LLM_API_KEYS=YOUR_GEMINI_API_KEY
LLM_MODEL=gemini-2.5-flash
LLM_VLM_MODEL=gemini-3.1-pro-preview
LLM_REWRITE_TEMPERATURE=0
LLM_PLANNER_TEMPERATURE=0
VLM_CANDIDATE_LIMIT=40
VLM_WEIGHT=0.65

AIC_ARTIFACT_DIR=video_search_bundle/clip-b32-btc-v1
AIC_CLIP_MODEL_NAME=ViT-B/32
AIC_USE_LORA=true
AIC_DEVICE=cuda
AIC_KEYFRAMES_ROOT=Keyframes
```

Có thể khai báo nhiều key, phân tách bằng dấu phẩy:

```env
LLM_API_KEYS=KEY_1,KEY_2,KEY_3
```

Không commit `.env` và không ghi API key vào source code.

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

## 6. Bật Gemini VLM

VLM dùng ảnh keyframe để rerank KIS hoặc trả lời QA. Đặt ảnh theo cấu trúc:

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
nếu dùng lại file đó, bước 6 sẽ bỏ qua query và không gọi Gemini. Kết quả VLM nằm trong
`outputs/results-vlm/`.

`LLM_VLM_MODEL` được tách khỏi model rewrite/planner. `VLM_CANDIDATE_LIMIT` kiểm soát
số keyframe gửi trong một lần rerank; `VLM_WEIGHT` là trọng số confidence của VLM khi
trộn với thứ hạng CLIP/RRF. Model Pro cho chất lượng tốt hơn nhưng thường chậm và tốn
quota hơn Flash.

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

- `LLM_API_KEYS is required`: chưa điền API key trong `.env`.
- `Missing CLIP manifest`: sai `AIC_ARTIFACT_DIR`.
- `LoRA manifest requires ...`: kiểm tra `AIC_USE_LORA=true` và `lora_weights.pt`.
- VLM không tìm thấy ảnh: kiểm tra `AIC_KEYFRAMES_ROOT` và đường dẫn keyframe.
- Lỗi thiết bị: kiểm tra `.env` có `AIC_DEVICE=cuda`, driver NVIDIA và PyTorch CUDA; chạy lại lệnh kiểm tra CUDA ở mục 2.
- Không có output: kiểm tra tên file có hậu tố `-kis`, `-qa` hoặc `-trake`.
