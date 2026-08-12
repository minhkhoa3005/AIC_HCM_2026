"""pipeline_batch_run.py

Chạy toàn bộ pipeline tự động:
1. Huấn luyện LoRA CLIP trên batch L21 (1,000 keyframes) dùng GPU (nếu có).
2. Trích xuất lại vector đặc trưng 512d với LoRA-CLIP.
3. Đẩy vector và metadata lên Qdrant Vector Database.
4. Truy vấn kết quả tìm kiếm với query = 'a photo of a tree', lưu ảnh keyframe kết quả và xuất báo cáo Markdown.
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import (
    BTC_CLIP_FEATURES_DIR,
    CLIP_MODEL_NAME,
    DEVICE,
    KEYFRAMES_DIR,
    LORA_ALPHA,
    LORA_RANK,
    LORA_WEIGHTS_PATH,
    METADATA_PATH,
    QDRANT_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    USE_REMOTE_VECTOR_DB,
)
from backend.embedding.clip_encoder import encode_image, encode_text_raw
from backend.training.lora import (
    count_trainable_params,
    inject_lora,
    load_lora_weights,
    save_lora_weights,
)
from scripts.train_lora_clip import (
    KeyframeCaptionDataset,
    _load_training_data,
    compute_clip_loss,
    train_one_epoch,
    validate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def step1_train_lora(device=None, limit: int = 1000, epochs: int = 3, batch_size: int = 32):
    """Bước 1: Huấn luyện LoRA CLIP (hỗ trợ NVIDIA CUDA & AMD DirectML GPU)."""
    if device is None:
        device = DEVICE
    logger.info("=== BƯỚC 1: HUẤN LUYỆN LORA CLIP TRÊN GPU/CPU ===")
    import clip

    dev_str = str(device)
    is_gpu = dev_str.startswith("cuda") or "privateuseone" in dev_str
    logger.info("Nạp CLIP %s trên thiết bị: %s (GPU Active: %s)", CLIP_MODEL_NAME, device, is_gpu)

    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)

    n_injected = inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    if LORA_WEIGHTS_PATH.exists():
        logger.info("Kế thừa tiến độ: Nạp checkpoint LoRA đã train từ trước từ %s", LORA_WEIGHTS_PATH)
        load_lora_weights(model, LORA_WEIGHTS_PATH)

    trainable, total = count_trainable_params(model)
    logger.info(
        "LoRA injected into %d layers. Trainable: %s / %s (%.2f%%)",
        n_injected, f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    items = _load_training_data(METADATA_PATH, limit=limit)
    if not items:
        logger.error("Không tìm thấy metadata!")
        return model, preprocess, device

    from torch.utils.data import DataLoader
    dataset = KeyframeCaptionDataset(items, preprocess, clip.tokenize)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, epoch, epochs)
        logger.info("Epoch %d/%d | Loss: %.4f", epoch, epochs, loss)

    save_lora_weights(
        model,
        LORA_WEIGHTS_PATH,
        metadata={"epochs": epochs, "limit": limit, "device": device},
    )
    logger.info("Hoàn tất lưu LoRA weights: %s", LORA_WEIGHTS_PATH)
    return model, preprocess, device


def step2_extract_features(model, preprocess, device: str, limit: int = 1000):
    """Bước 2: Trích xuất vector đặc trưng bằng LoRA-CLIP."""
    logger.info("=== BƯỚC 2: TRÍCH XUẤT VECTOR BẰNG LORA-CLIP ===")
    items = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            if line.strip():
                items.append(json.loads(line))

    model.eval()
    extracted_count = 0
    with torch.no_grad():
        for item in items:
            img_path = Path(item["path"])
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img_t = preprocess(img).unsqueeze(0).to(device)
                feat = model.encode_image(img_t)
                feat = (feat / feat.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze(0)

                video_id = item["video_id"]
                frame_stem = img_path.stem
                batch_name = video_id.split("_")[0]

                out_dir = BTC_CLIP_FEATURES_DIR / batch_name / video_id
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"{frame_stem}.npy", feat)
                extracted_count += 1
            except Exception as e:
                logger.warning("Lỗi trích xuất %s: %s", img_path, e)

    logger.info("Đã trích xuất %d vectors đặc trưng mới.", extracted_count)


def step3_push_to_qdrant(limit: int = 1000):
    """Bước 3: Đẩy vectors và metadata lên Qdrant Database."""
    logger.info("=== BƯỚC 3: ĐẨY VECTORS LÊN QDRANT DATABASE ===")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from backend.config import INDEX_DIR, QDRANT_URL, QDRANT_API_KEY

    try:
        if QDRANT_URL:
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        else:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
            client.get_collections()
        logger.info("Kết nối tới Qdrant Server thành công.")
    except Exception as e:
        qdrant_db_path = str(INDEX_DIR / "qdrant_db")
        logger.warning("Không thể kết nối Qdrant Daemon (%s). Chuyển sang Qdrant Local Storage tại '%s'...", e, qdrant_db_path)
        client = QdrantClient(path=qdrant_db_path)

    # Recreate collection
    try:
        if client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
            client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        logger.info("Tạo mới Qdrant collection: %s", QDRANT_COLLECTION_NAME)
    except Exception as e:
        logger.warning("Thử tạo collection: %s", e)

    items = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            if line.strip():
                items.append(json.loads(line))

    points = []
    for idx, item in enumerate(items):
        video_id = item["video_id"]
        frame_stem = Path(item["path"]).stem
        batch_name = video_id.split("_")[0]
        feat_path = BTC_CLIP_FEATURES_DIR / batch_name / video_id / f"{frame_stem}.npy"

        if feat_path.exists():
            vec = np.load(feat_path).tolist()
            payload = {
                "video_id": video_id,
                "frame_id": item.get("frame_id", 0),
                "pts_time": item.get("pts_time", 0.0),
                "path": item.get("path", ""),
                "caption": item.get("caption", ""),
                "text": item.get("text", ""),
            }
            points.append(PointStruct(id=idx + 1, vector=vec, payload=payload))

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
        logger.info("Đã đẩy %d vectors lên Qdrant thành công!", len(points))
    return client


def step4_query_and_generate_artifact(client, query_text: str = "a photo of a tree"):
    """Bước 4: Truy vấn tìm kiếm, trích xuất ảnh và tạo báo cáo Markdown."""
    logger.info("=== BƯỚC 4: TRUY VẤN TÌM KIẾM & XUẤT BÁO CÁO ===")
    query_vec = encode_text_raw(query_text)

    try:
        res = client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vec.tolist(),
            limit=5,
        )
        search_results = res.points
    except Exception as e:
        logger.warning("Truy vấn với query_points thất bại (%s), thử lại với API tương thích...", e)
        search_results = client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=5)[0]

    artifact_dir = Path("C:/Users/Administrator/.gemini/antigravity-ide/brain/c606a4f6-beb2-4fa0-a0f2-46a8ba7ee5b1")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    md_content = f"# Kết quả Tìm Kiếm sau khi Fine-tune LoRA CLIP\n\n"
    md_content += f"- **Query Text**: `{query_text}`\n"
    md_content += f"- **Vector Dim**: 512d\n"
    md_content += f"- **Model**: CLIP ViT-B/32 (LoRA-adapted)\n\n"
    md_content += f"| Rank | Score | Video ID | Frame ID | Timestamp | Ảnh Khoảnh Khắc |\n"
    md_content += f"| :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for rank, hit in enumerate(search_results, 1):
        p = hit.payload
        vid = p.get("video_id", "N/A")
        fid = p.get("frame_id", 0)
        pts = p.get("pts_time", 0.0)
        img_src_path = Path(p.get("path", ""))

        img_artifact_name = f"{vid}_{fid}.jpg"
        img_artifact_path = artifact_dir / img_artifact_name

        if img_src_path.exists():
            try:
                img = Image.open(img_src_path)
                img.save(img_artifact_path)
            except Exception as e:
                logger.warning("Không thể lưu ảnh artifact: %s", e)

        md_content += f"| **#{rank:02d}** | `{hit.score:.4f}` | `{vid}` | `{fid}` | `{pts:.1f}s` | ![{vid}_{fid}]({img_artifact_path.as_uri()}) |\n"

    md_file = artifact_dir / "search_results.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Đã lưu kết quả truy vấn và ảnh báo cáo tại: %s", md_file)
    print("\n" + md_content)


def main():
    device = DEVICE  # GPU AMD DirectML mặc định cho cả Training + Inference
    batch_size = 32
    epochs = 3
    limit = 0  # 0 = FULL toàn bộ 22,248 keyframes của Video_L21

    logger.info("=== HUẤN LUYỆN LORA CLIP FULL L21 (22,248 KEYFRAMES) trên %s ===", device)

    # 1. Train LoRA trên GPU AMD DirectML
    model, preprocess, _ = step1_train_lora(device=device, limit=limit, epochs=epochs, batch_size=batch_size)

    # 2. Trích xuất đặc trưng FULL 22,248 keyframe bằng GPU AMD DirectML
    step2_extract_features(model, preprocess, device=device, limit=limit)

    # 3. Push Qdrant FULL 22,248 keyframe
    client = step3_push_to_qdrant(limit=limit)

    # 4. Search & Output Artifact (Score + Images)
    step4_query_and_generate_artifact(client, query_text="một bức ảnh về cái cây")


if __name__ == "__main__":
    main()
