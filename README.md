# AIC 2026 — Video Search Agent

Hệ thống tìm kiếm khoảnh khắc video đa phương thức (Multimodal Video Moment Retrieval) cho cuộc thi AIC 2026.  
Kiến trúc **keyframe-centric**: trả về `video_id`, `frame_id` và dòng submission có thể copy trực tiếp.

---

## 🌟 Điểm Nổi Bật Kỹ Thuật

1. **LoRA Fine-Tuning cho CLIP Backbone**:
   - Fine-tune CLIP ViT-B/32 chống **Catastrophic Forgetting** (chỉ ~0.28% tham số được train, checkpoint chỉ ~1.7 MB).
   - Tự động nạp bộ gia tăng LoRA (`lora_weights.pt`) khi trích xuất hoặc tìm kiếm.
   - Hỗ trợ học tăng cường (`--resume`) theo từng đợt dữ liệu (batch/video).
2. **Hỗ Trợ GPU / CPU Linh Hoạt**:
   - Tùy chỉnh thiết bị qua dòng lệnh (`--device cuda`, `--device cpu`, `--device cuda:0`) hoặc qua file `.env` (`DEVICE=cuda`).
3. **Tích Hợp Âm Thanh (Whisper ASR) & Hình Ảnh**:
   - Trích xuất transcript lời thoại từ audio video và gán trực tiếp vào timestamp keyframe.
4. **Local FAISS Index**:
   - Quản lý và tìm kiếm similarity vector siêu tốc với local FAISS.

---

## 🎯 Dạng Truy Vấn Hỗ Trợ

| Dạng bài | Output format |
|----------|---------------|
| Textual KIS | `<video_id>, <frame_id>` |
| Q&A | `<video_id>, <frame_id>, <answer>` |
| TRAKE | `<video_id>, <frame_id_1>, ..., <frame_id_n>` |

Mỗi truy vấn lấy tối đa Top 100 kết quả — phù hợp cách chấm `R@1`, `R@5`, `R@20`, `R@50`, `R@100`.

---

## 🏗️ Kiến Trúc Thư Mục

```text
video-search-agent/
├── ZIP/                          # File zip dữ liệu BTC gốc
├── data/
│   ├── keyframes/                # Ảnh keyframe (từ BTC)
│   ├── clip-features/            # CLIP feature vectors .npy (từ BTC)
│   ├── map-keyframes/            # Mapping frame_id (từ BTC)
│   ├── media-info/               # FPS, duration (từ BTC)
│   ├── objects/                  # Object detection tags (từ BTC)
│   └── index/                    # Output: metadata.jsonl, lora_weights.pt
├── backend/
│   ├── config.py                 # Cấu hình trung tâm (ENV, DEVICE, paths)
│   ├── preprocessing/            # Import metadata, Whisper transcribe, generate captions
│   ├── training/                 # LoRA module, Projection Head & Temporal GRU
│   └── embedding/                # CLIP encoder (LoRA-aware)
├── scripts/                      # CLI scripts chạy pipeline & training
│   ├── extract_btc_data.py
│   ├── import_btc_data.py
│   ├── train_lora_clip.py       # Fine-tune CLIP bằng LoRA
│   ├── extract_clip_features.py  # Trích xuất vector (dùng LoRA nếu có)
│   └── build_index_features.py
├── run_pipeline.ipynb            # Notebook tự động chạy toàn bộ pipeline
├── search_demo.ipynb             # Notebook demo tìm kiếm & trực quan kết quả
└── requirements.txt
```

---

## ⚙️ Cài Đặt & Cấu Hình

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình GPU / CPU (`.env`)

Tạo hoặc chỉnh sửa file `.env` ở thư mục gốc:

```env
# Cấu hình thiết bị (cuda | cuda:0 | cpu)
DEVICE=cuda

# Cấu hình Whisper ASR
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Cấu hình LoRA
LORA_RANK=4
LORA_ALPHA=1.0
USE_LORA=auto

# FAISS Local Index
USE_REMOTE_VECTOR_DB=false
```

---

## 📓 Chạy Qua Jupyter Notebook

Mở notebook hướng dẫn end-to-end:

```bash
jupyter notebook run_pipeline.ipynb
```

Hoặc thử nghiệm tìm kiếm trực quan:

```bash
jupyter notebook search_demo.ipynb
```

---

## 💻 Pipeline Chạy Bằng Command Line

### Bước 1 — Giải nén dữ liệu BTC

Đặt các file `.zip` từ ban tổ chức vào thư mục `ZIP/`, sau đó giải nén:

```bash
python scripts/extract_btc_data.py
```

### Bước 2 — Import Metadata & Audio Transcripts

Tạo `metadata.jsonl` từ dữ liệu keyframes và transcript âm thanh:

```bash
python scripts/import_btc_data.py --with-transcript
```

### Bước 3 — Fine-tune CLIP bằng LoRA (Chống Catastrophic Forgetting)

Huấn luyện bộ adapter LoRA trên dữ liệu AIC (hỗ trợ GPU/CPU và học tăng cường `--resume`):

```bash
# Huấn luyện trên GPU với 5 epochs
python scripts/train_lora_clip.py --device cuda --epochs 5 --batch-size 32

# Học tiếp (incremental learning) từ checkpoint cũ
python scripts/train_lora_clip.py --device cuda --resume --epochs 3
```

Output: `data/index/lora_weights.pt` (~1.7 MB).

### Bước 4 — Trích xuất Features (Dùng LoRA-CLIP)

Chiết xuất lại vector đặc trưng cho keyframes sử dụng model đã fine-tune:

```bash
python scripts/extract_clip_features.py --device cuda
```

### Bước 5 — Build FAISS Local Index

Xây dựng FAISS Index từ dữ liệu:
python backend/embedding/push_to_remote.py --recreate

# Hoặc build local FAISS index
python scripts/build_index_features.py --remote
```

---

## 🔬 Kiểm Thử Tìm Kiếm

Chạy thử tìm kiếm từ dòng lệnh:

```python
from backend.embedding.clip_encoder import encode_text_raw

# Encode query text bằng LoRA-CLIP
query_vec = encode_text_raw("a photo of a tree")
```

---

## 📝 Giấy Phép & Đóng Góp

Dự án phát triển cho cuộc thi **Ho Chi Minh City AI Challenge (AIC) 2026**.
