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
    resolve_path,
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


from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm

class KeyframeDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            from PIL import Image
            img = Image.open(str(path)).convert("RGB")
            img_t = self.preprocess(img)
            return img_t, str(path), True
        except Exception:
            return torch.zeros((3, 224, 224)), str(path), False

def step2_extract_features(model, preprocess, device: str, limit: int = 0):
    """Bước 2: Trích xuất vector đặc trưng bằng LoRA-CLIP."""
    logger.info("=== BƯỚC 2: TRÍCH XUẤT VECTOR BẰNG LORA-CLIP ===")

    if not KEYFRAMES_DIR.exists():
        logger.error("Thư mục keyframes không tồn tại: %s", KEYFRAMES_DIR)
        return

    image_paths = sorted(KEYFRAMES_DIR.rglob("*.jpg"))
    if limit > 0:
        image_paths = image_paths[:limit]

    if not image_paths:
        logger.warning("Không tìm thấy file .jpg nào trong %s", KEYFRAMES_DIR)
        return

    paths_to_process = []
    for img_path in image_paths:
        video_id = img_path.parent.name
        frame_stem = img_path.stem
        save_path = BTC_CLIP_FEATURES_DIR / video_id / f"{frame_stem}.npy"
        if not save_path.exists():
            paths_to_process.append(img_path)

    if not paths_to_process:
        logger.info("Tất cả %d keyframes đã được trích xuất (resume success).", len(image_paths))
        return

    logger.info("Cần trích xuất đặc trưng cho %d keyframes mới...", len(paths_to_process))
    model.eval()
    
    dataset = KeyframeDataset(paths_to_process, preprocess)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    extracted_count = 0
    with torch.no_grad():
        with tqdm(total=len(paths_to_process), desc="Extracting Features") as pbar:
            for imgs, paths, valids in dataloader:
                valid_mask = valids.numpy()
                if not valid_mask.any():
                    pbar.update(len(paths))
                    continue
                
                valid_imgs = imgs[valids].to(device)
                feat = model.encode_image(valid_imgs).float()
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embs = feat.cpu().numpy()
                
                emb_idx = 0
                for i, path_str in enumerate(paths):
                    if valids[i]:
                        img_path = Path(path_str)
                        video_id = img_path.parent.name
                        frame_stem = img_path.stem
                        
                        out_dir = BTC_CLIP_FEATURES_DIR / video_id
                        out_dir.mkdir(parents=True, exist_ok=True)
                        save_path = out_dir / f"{frame_stem}.npy"
                        
                        np.save(save_path, embs[emb_idx])
                        emb_idx += 1
                        extracted_count += 1
                
                pbar.update(len(paths))

    logger.info("Đã trích xuất %d vectors đặc trưng mới.", extracted_count)



def step3_build_faiss_index(limit: int = 1000):
    """Bước 3: Xây dựng FAISS Index Local."""
    logger.info("=== BƯỚC 3: XÂY DỰNG FAISS INDEX LOCAL ===")
    import faiss
    from backend.config import INDEX_DIR, FAISS_INDEX_PATH, FAISS_METADATA_PATH

    items = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            if line.strip():
                items.append(json.loads(line))

    vectors = []
    metadata = []
    dim = 512

    def load_feat(item_and_idx):
        idx, item = item_and_idx
        video_id = item["video_id"]
        frame_stem = resolve_path(item.get("path", "")).stem
        batch_name = video_id.split("_")[0]
        
        candidates = [
            BTC_CLIP_FEATURES_DIR / batch_name / video_id / f"{frame_stem}.npy",
            BTC_CLIP_FEATURES_DIR / video_id / f"{frame_stem}.npy"
        ]
        
        feat_path = next((p for p in candidates if p.exists()), None)
        if feat_path:
            vec = np.load(feat_path)
            payload = {
                "id": idx,
                "video_id": video_id,
                "frame_id": item.get("frame_id", 0),
                "pts_time": item.get("pts_time", 0.0),
                "path": item.get("path", ""),
                "caption": item.get("caption", ""),
                "text": item.get("text", ""),
            }
            return vec, payload
        return None, None

    import concurrent.futures
    logger.info("Nạp %d features bằng đa luồng...", len(items))
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(load_feat, enumerate(items)))

    for vec, payload in results:
        if vec is not None:
            vectors.append(vec)
            metadata.append(payload)

    if not vectors:
        logger.warning("Không tìm thấy vector nào để đưa vào FAISS.")
        return None, None

    vectors_np = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(vectors_np)
    
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)
    
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    logger.info("Đã xây dựng xong FAISS Index (%d chiều) với %d keyframes.", dim, len(metadata))
    return index, metadata


def step4_query_and_generate_artifact(index, metadata, query_text: str = "a photo of a tree"):
    """Bước 4: Truy vấn tìm kiếm (MMR & Rocchio & Temporal), trích xuất ảnh và tạo báo cáo Markdown."""
    logger.info("=== BƯỚC 4: TRUY VẤN TÌM KIẾM (MMR & ROCCHIO & TEMPORAL) & XUẤT BÁO CÁO ===")
    import faiss
    from backend.embedding.search_algorithms import mmr_search, rocchio_feedback, temporal_search
    from backend.embedding.clip_encoder import encode_text_raw, translate_vi_to_en

    if index is None:
        logger.error("FAISS Index is None!")
        return

    top_k = 5
    artifact_dir = Path("search_results").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def encode_fn(text):
        text_en = translate_vi_to_en(text)
        if text_en != text:
            logger.info("Translated sub-query: '%s' -> '%s'", text, text_en)
        query_vec = encode_text_raw(text_en)
        query_super = query_vec.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_super)
        return query_super

    # 4.1 Thực hiện tìm kiếm với Temporal Search (Tự động fallback về MMR nếu không có dấu mũi tên)
    logger.info("Đang thực hiện tìm kiếm Temporal / MMR...")
    scores_mmr, results_mmr, sub_queries = temporal_search(
        temporal_query_text=query_text,
        encode_fn=encode_fn,
        index=index,
        metadata=metadata,
        max_gap_sec=120.0,
        top_k_candidates=50,
        top_k=top_k
    )

    is_temporal = len(sub_queries) > 1

    # 4.2 Giả lập Rocchio Feedback (Chỉ áp dụng nếu KHÔNG PHẢI Temporal)
    scores_rocchio, results_rocchio = scores_mmr, results_mmr
    if not is_temporal:
        # Giả sử người dùng phản hồi kết quả đầu tiên là đúng (Positive), kết quả cuối cùng là sai (Negative)
        logger.info("Đang thực hiện Rocchio Feedback (1 Positive, 1 Negative)...")
        if len(results_mmr) >= 2:
            rel_vecs = []
            non_rel_vecs = []
            try:
                rel_vecs.append(index.reconstruct(int(results_mmr[0]['id'])))
                non_rel_vecs.append(index.reconstruct(int(results_mmr[-1]['id'])))

                # Lấy vector query ban đầu
                original_query = encode_fn(query_text)

                # Cập nhật query (alpha=1.0, beta=0.75, gamma=0.15)
                new_query = rocchio_feedback(original_query, rel_vecs, non_rel_vecs, alpha=1.0, beta=0.75, gamma=0.15)

                # Thực hiện lại tìm kiếm sau khi cập nhật
                scores_rocchio, results_rocchio = mmr_search(new_query, index, metadata, top_k=top_k, lambda_mult=0.5, fetch_k=50)
                logger.info("Đã tìm kiếm lại với vector từ Rocchio Feedback.")
            except Exception as e:
                logger.warning("Rocchio feedback gặp lỗi (có thể index không hỗ trợ reconstruct): %s", e)

    def render_table(title, scores, results, is_temp=False):
        content = f"## {title}\n\n"
        if is_temp:
            content += f"| Rank | Score | Video ID | Frame ID | Thời điểm A -> B | Gap (s) | Ảnh Khoảnh Khắc B |\n"
            content += f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        else:
            content += f"| Rank | Score | Video ID | Frame ID | Timestamp | Ảnh Khoảnh Khắc |\n"
            content += f"| :---: | :---: | :---: | :---: | :---: | :---: |\n"
        
        for i, (score, p) in enumerate(zip(scores, results)):
            vid = p.get('video_id', 'N/A')
            fid = p.get('frame_id', 0)
            pts = p.get('pts_time', 0.0)
            img_src_path = Path(p.get('path', ''))
            img_artifact_path = artifact_dir / f"{vid}_{fid}.jpg"

            if img_src_path.exists():
                try:
                    img = Image.open(img_src_path)
                    img.save(img_artifact_path)
                except Exception:
                    pass
            
            if is_temp:
                gap = p.get('time_gap', 0.0)
                pts_a = p.get('event_a_pts', 0.0)
                content += f"| **#{i+1:02d}** | `{score:.4f}` | `{vid}` | `{fid}` | `{pts_a:.1f}s -> {pts:.1f}s` | `{gap:.1f}s` | ![{vid}_{fid}]({img_artifact_path.as_uri()}) |\n"
            else:
                content += f"| **#{i+1:02d}** | `{score:.4f}` | `{vid}` | `{fid}` | `{pts:.1f}s` | ![{vid}_{fid}]({img_artifact_path.as_uri()}) |\n"
        return content

    md_content = f"# Kết quả Tìm Kiếm sau khi Fine-tune LoRA CLIP (Nâng Cao)\n\n"
    md_content += f"- **Query Text**: `{query_text}`\n"
    if is_temporal:
        md_content += f"- **Sub-Queries**: `{'` -> `'.join(sub_queries)}`\n\n"
        md_content += render_table("1. Tìm kiếm với Temporal Search", scores_mmr, results_mmr, is_temp=True)
    else:
        query_en = translate_vi_to_en(query_text)
        md_content += f"- **Translated**: `{query_en}`\n\n"
        md_content += render_table("1. Tìm kiếm với MMR (Đa dạng hóa, lambda=0.5)", scores_mmr, results_mmr)
        md_content += render_table("2. Tìm kiếm với Rocchio Feedback (Sau khi cập nhật Vector)", scores_rocchio, results_rocchio)

    md_file = artifact_dir / "search_results.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Đã lưu kết quả truy vấn và ảnh báo cáo tại: %s", md_file)
    print(f"\n[Success] Advanced Markdown report saved at: {md_file}")


def main():
    train_device = DEVICE  # Dùng thiết bị mặc định (GPU/DML) thay vì ép CPU
    gpu_device = DEVICE
    batch_size = 4  # Giảm batch_size xuống 4 để tránh tràn RAM GPU AMD
    epochs = 3
    limit = 0  # Chỉ dùng 10 keyframes để test thử nhanh

    logger.info("=== HUẤN LUYỆN LORA CLIP FULL L21 (1000 KEYFRAMES) ===")

    # 1. Train LoRA trên CPU
    model, preprocess, _ = step1_train_lora(device=train_device, limit=limit, epochs=epochs, batch_size=batch_size)

    # Chuyển mô hình sang GPU trước khi trích xuất đặc trưng (vì quá trình train đang dùng CPU)
    logger.info("Chuyển model sang %s để trích xuất ảnh...", gpu_device)
    model = model.to(gpu_device)

    # 2. Trích xuất đặc trưng bằng GPU
    step2_extract_features(model, preprocess, device=gpu_device, limit=limit)

    # 3. Build FAISS Index
    index, metadata = step3_build_faiss_index(limit=limit)

    # 4. Search & Output Artifact (Score + Images)
    logger.info("=== BƯỚC 4: THỬ NGHIỆM TÌM KIẾM MẪU ===")
    queries = [
        "a photo of a person wearing a red shirt",
        "car moving on the highway"
    ]
    for q in queries:
        step4_search_and_output(q, index, metadata)

    step4_query_and_generate_artifact(index, metadata, query_text="bản tin thời sự -> buổi sáng")


if __name__ == "__main__":
    main()
