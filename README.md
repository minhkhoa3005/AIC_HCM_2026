# AIC 2026 — Video Search Agent

Hệ thống tìm kiếm khoảnh khắc video cho vòng sơ tuyển AIC 2026.  
Kiến trúc **keyframe-centric**: trả về `video_id`, `frame_id` và dòng submission có thể copy trực tiếp.

## Dạng Truy Vấn Hỗ Trợ

| Dạng bài | Output format |
|----------|---------------|
| Textual KIS | `<video_id>, <frame_id>` |
| Q&A | `<video_id>, <frame_id>, <answer>` |
| TRAKE | `<video_id>, <frame_id_1>, ..., <frame_id_n>` |

Mỗi truy vấn lấy tối đa Top 100 kết quả — phù hợp cách chấm `R@1`, `R@5`, `R@20`, `R@50`, `R@100`.

## Kiến Trúc Tổng Quan

```text
ZIP/*.zip (dữ liệu BTC)
  ↓ extract_btc_data.py
data/keyframes/ + clip-features/ + map-keyframes/ + media-info/ + objects/
  ↓ import_btc_data.py
data/index/metadata.jsonl (keyframe metadata đầy đủ)
  ↓ train.py + train_temporal.py
data/index/projection_head.pt + temporal_encoder.pt
  ↓ push_to_remote.py
Qdrant Remote Vector DB (search-ready)
```

## Cấu Trúc Thư Mục

```
video-search-agent/
├── ZIP/                          # File zip dữ liệu BTC gốc
├── data/
│   ├── keyframes/                # Ảnh keyframe (từ BTC)
│   ├── clip-features/            # CLIP feature vectors .npy (từ BTC)
│   ├── map-keyframes/            # Mapping frame_id (từ BTC)
│   ├── media-info/               # FPS, duration (từ BTC)
│   ├── objects/                  # Object detection tags (từ BTC)
│   └── index/                    # Output: metadata, model checkpoints
├── backend/
│   ├── config.py                 # Cấu hình trung tâm
│   ├── preprocessing/            # Import & xử lý dữ liệu BTC
│   ├── training/                 # Huấn luyện Projection Head & Temporal GRU
│   └── embedding/                # CLIP encoder, push lên Remote Vector DB
├── scripts/                      # CLI scripts chạy pipeline
└── frontend/                     # Giao diện web
```

---

## Cài Đặt

### Yêu cầu

- Python >= 3.10
- GPU (khuyến nghị cho training, không bắt buộc)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

---

## 📓 Chạy Nhanh Qua Jupyter Notebook

Nếu bạn muốn chạy từng bước trực quan kèm tự động giải nén và kiểm tra kết quả, mở file Notebook:

```bash
jupyter notebook run_pipeline.ipynb
```

File [run_pipeline.ipynb](file:///d:/AI/AIC%202026/video-search-agent/run_pipeline.ipynb) bao gồm đầy đủ từ giải nén zip, import metadata, train Projection Head, train Temporal GRU đến push Qdrant Remote và test query.

---

## 💻 Pipeline Chạy Bằng Command Line

### Bước 0 — Chuẩn bị dữ liệu BTC

Đặt các file `.zip` từ ban tổ chức vào thư mục `ZIP/`:

```
ZIP/
├── Keyframes_L21.zip
├── clip-features-32-aic25-b1.zip
├── map-keyframes-aic25-b1.zip
├── media-info-aic25-b1.zip
└── objects-aic25-b1.zip
```

### Bước 1 — Giải nén dữ liệu

Script tự động mapping tên thư mục trong zip vào đúng thư mục đích:

```bash
# Xem trước (không giải nén)
python scripts/extract_btc_data.py --dry-run

# Giải nén thực
python scripts/extract_btc_data.py
```

### Bước 2 — Import metadata

Quét dữ liệu BTC, tạo `metadata.jsonl` với đầy đủ `frame_id`, `pts_time`, object tags:

```bash
python scripts/import_btc_data.py
```

Output: `data/index/metadata.jsonl`

### Bước 3 — Huấn luyện Projection Head

Fine-tune hai Projection Head (image + text) bằng InfoNCE contrastive loss:

```bash
python -m backend.training.train
```

Output: `data/index/projection_head.pt`

**Chi tiết:**
- Dữ liệu: raw CLIP features (512d) + CLIP text encoding từ caption/metadata
- Model: `ProjectionHead` (Linear → GELU → Linear), 512d → 256d
- Loss: InfoNCE hai chiều + Learnable Temperature
- Anti-false-negative: `UniqueVideoBatchSampler` (không trùng video trong batch)
- Validation Loss mỗi epoch để theo dõi overfitting

### Bước 4 — Huấn luyện Temporal Video Encoder

Fine-tune GRU 1-layer encode chuỗi keyframes thành video-level representation:

```bash
python -m backend.training.train_temporal
```

Output: `data/index/temporal_encoder.pt`

**Chi tiết:**
- Dữ liệu: chuỗi keyframe features gom theo video
- Model: `TemporalVideoEncoder` (GRU → Mean Pooling → Linear), 512d → 256d
- Tự động padding sequences có độ dài khác nhau trong batch

### Bước 5 — Push vectors lên Remote (Qdrant)

```bash
# Push raw CLIP features (512d) — khuyến nghị
python backend/embedding/push_to_remote.py

# Hoặc push projected vectors (256d) nếu đã train Projection Head
python backend/embedding/push_to_remote.py --projected

# Tạo lại collection từ đầu
python backend/embedding/push_to_remote.py --recreate
```

### Bước 6 (Tuỳ chọn) — Build FAISS local index

Nếu muốn lưu index local (thay vì Remote):

```bash
python scripts/build_index_features.py

# Hoặc build local + push remote cùng lúc
python scripts/build_index_features.py --remote
```

---

## Cấu Hình

Toàn bộ cấu hình nằm trong `backend/config.py`:

| Tham số | Default | Mô tả |
|---------|---------|-------|
| `CLIP_MODEL_NAME` | `ViT-B/32` | CLIP backbone |
| `EMBED_DIM` | `512` | Chiều vector CLIP gốc |
| `PROJECTED_DIM` | `256` | Chiều vector sau Projection Head |
| `TRAIN_BATCH_SIZE` | `32` | Batch size training |
| `TRAIN_EPOCHS` | `8` | Số epoch training |
| `TRAIN_LR` | `1e-4` | Learning rate |
| `TEMPERATURE` | `0.07` | Initial temperature (learnable) |
| `SMART_CUT_SIMILARITY_THRESHOLD` | `0.82` | Ngưỡng cosine similarity cắt scene |

### Remote Vector DB (Qdrant)

Cấu hình qua biến môi trường:

```bash
export USE_REMOTE_VECTOR_DB=true
export QDRANT_URL=https://your-cluster.qdrant.io
export QDRANT_API_KEY=your-api-key
export QDRANT_COLLECTION_NAME=aic2026_keyframes
```

---

## Ghi Chú Theo Đề Sơ Tuyển

- **Dữ liệu chính thức** để chấm là video, nhưng pipeline dùng keyframes + CLIP features + metadata do BTC cung cấp để định vị đúng `frame_id`.
- **VFR (Variable Frame Rate):** Ưu tiên `frame_id`/timestamp từ `map-keyframes`; chỉ fallback sang tên file/fps khi metadata BTC thiếu.
- **Q&A:** Hệ thống hỗ trợ định vị khoảnh khắc và để người dùng điền answer sau khi xem lại.
- **TRAKE:** Hệ thống chọn keyframe tốt nhất theo scene và sắp xếp các `frame_id` theo thứ tự thời gian trong submission.
