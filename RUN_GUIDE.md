# Hướng dẫn chạy hệ thống LLM

Tài liệu này hướng dẫn chạy độc lập luồng KIS, QA và TRAKE bằng bundle CLIP/FAISS local.

## 1. Cấu trúc thư mục

```text
LLM/
├── .env
├── run_inference.py
├── inputs/
├── outputs/
├── answer/
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
LLM_REWRITE_TEMPERATURE=0
LLM_PLANNER_TEMPERATURE=0

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
├── query-1-kis-result.csv
├── query-2-qa-result.csv
└── query-4-trake-result.csv
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

TRAKE hiện dùng CLIP/FAISS và dynamic programming để giữ thứ tự frame; VLM không bắt buộc.

## 7. Chuẩn bị đáp án đúng

Đặt đáp án BTC trong `answer/`, cùng query stem với file input:

```text
answer/
├── query-1-kis.txt
├── query-2-qa.txt
└── query-4-trake.txt
```

KIS/QA:

```text
Video ID: L21_V001
Timestamp: 03:46 - 03:50
Frame Range: [5650, 5750]
```

QA thêm:

```text
Answer: màu xanh
```

TRAKE:

```text
Video ID: L21_V001
Event 1: [6700, 6750]
Event 2: [6775, 6825]
Event 3: [7425, 7475]
```

Frame range được tính bao gồm cả hai đầu mút.

## 8. Chấm điểm local

```powershell
python answer/evaluate.py `
  --answer-dir answer `
  --result-dir outputs/results `
  --report outputs/evaluation_report.json
```

Evaluator ghi:

```text
R@1
R@5
R@20
R@50
R@100
Final Score
```

Ngoài ra báo cáo có `first_hit_rank`, `correct_ranks` và `correct_predictions` để biết frame đúng nằm ở Top nào.

Để chấm kết quả có VLM, đổi thư mục kết quả và tên báo cáo:

```powershell
python answer/evaluate.py `
  --answer-dir answer `
  --result-dir outputs/results-vlm `
  --report outputs/evaluation-vlm.json
```

## 9. Chế độ CSV cũ

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

## 10. Kiểm tra nhanh khi lỗi

- `LLM_API_KEYS is required`: chưa điền API key trong `.env`.
- `Missing CLIP manifest`: sai `AIC_ARTIFACT_DIR`.
- `LoRA manifest requires ...`: kiểm tra `AIC_USE_LORA=true` và `lora_weights.pt`.
- VLM không tìm thấy ảnh: kiểm tra `AIC_KEYFRAMES_ROOT` và đường dẫn keyframe.
- Lỗi thiết bị: kiểm tra `.env` có `AIC_DEVICE=cuda`, driver NVIDIA và PyTorch CUDA; chạy lại lệnh kiểm tra CUDA ở mục 2.
- Không có output: kiểm tra tên file có hậu tố `-kis`, `-qa` hoặc `-trake`.
