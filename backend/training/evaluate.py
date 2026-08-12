"""
Đánh giá độ chính xác thực tế (Realistic Recall@K & MRR) bằng cách đưa
toàn bộ Database/Distractor Pool vào không gian tìm kiếm.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

from backend.config import CLIP_MODEL_NAME
import numpy as np
import torch
import torch.nn.functional as F
from backend.embedding.clip_encoder import encode_texts_raw

from backend.config import (
    BTC_CLIP_FEATURES_DIR,
    CLIP_MODEL_NAME,
    DEVICE,
    METADATA_PATH,
    PROJECTION_HEAD_PATH,
)
from backend.training.dataset import build_pairs, train_val_split
from backend.training.model import ProjectionHead


def load_all_distractors() -> Tuple[List[str], np.ndarray]:
    """Load toàn bộ vector feature (.npy) có trong hệ thống để làm Distractor Pool."""
    distractor_ids = []
    distractor_feats = []

    print("Loading toàn bộ Distractor Pool từ đĩa...")
    for npy_file in BTC_CLIP_FEATURES_DIR.rglob("*.npy"):
        try:
            arr = np.load(npy_file).astype(np.float32)
            if arr.ndim == 1:
                item_id = f"{npy_file.parent.name}/{npy_file.stem}"
                distractor_ids.append(item_id)
                distractor_feats.append(arr)
            elif arr.ndim == 2:
                # File stacked theo video
                video_id = npy_file.stem
                for idx, row in enumerate(arr):
                    item_id = f"{video_id}/{idx:04d}"
                    distractor_ids.append(item_id)
                    distractor_feats.append(row)
        except Exception:
            continue

    if not distractor_feats:
        return [], np.empty((0, 512), dtype=np.float32)

    return distractor_ids, np.stack(distractor_feats).astype(np.float32)


def evaluate_with_distractors(
    val_pairs: List[Dict],
    top_k_list: List[int] = [1, 5, 10, 50],
):
    """
    Tính Recall@K và MRR trong không gian tìm kiếm thực tế (gồm cả Distractors).
    """
    if not val_pairs:
        print("Tập Validation rỗng.")
        return

    image_head, text_head = None, None
    if PROJECTION_HEAD_PATH.exists():
        ckpt = torch.load(PROJECTION_HEAD_PATH, map_location=DEVICE, weights_only=False)
        input_dim = ckpt.get("input_dim") or ckpt.get("hyperparameters", {}).get("input_dim", 512)
        projected_dim = ckpt.get("projected_dim") or ckpt.get("hyperparameters", {}).get("projected_dim", 256)
        image_head = ProjectionHead(input_dim, projected_dim).to(DEVICE)
        text_head = ProjectionHead(input_dim, projected_dim).to(DEVICE)
        image_head.load_state_dict(ckpt["image_head"])
        text_head.load_state_dict(ckpt["text_head"])
        image_head.eval()
        text_head.eval()

    # 2. Chuẩn bị Distractor Pool (Database)
    distractor_ids, distractor_matrix = load_all_distractors()
    if len(distractor_matrix) == 0:
        print("Không tìm thấy Distractors. Vui lòng kiểm tra lại đường dẫn features.")
        return

    # Transform Distractor Embeddings qua Projection Head nếu có
    with torch.no_grad():
        dist_tensor = torch.from_numpy(distractor_matrix).to(DEVICE)
        if image_head is not None:
            dist_tensor = image_head(dist_tensor)
        dist_tensor = F.normalize(dist_tensor, dim=-1)
        dist_matrix_norm = dist_tensor.cpu().numpy()

    # 3. Encoding Query Text & Lấy Ground Truth Index trong Distractor Pool
    val_texts = [p["train_text"] for p in val_pairs]
    gt_identifiers = [f"{p['video_id']}/{Path(p['feat_path']).stem}" for p in val_pairs]

    # Map Ground Truth ID sang index trong distractor_matrix
    id_to_index = {item_id: idx for idx, item_id in enumerate(distractor_ids)}
    valid_queries = []
    gt_indices = []

    for i, gt_id in enumerate(gt_identifiers):
        if gt_id in id_to_index:
            valid_queries.append(val_texts[i])
            gt_indices.append(id_to_index[gt_id])

    if not valid_queries:
        print("Không khớp được Ground Truth index nào trong Distractor Pool.")
        return

    print(f"Tổng số Query Validation thực tế: {len(valid_queries)}")
    print(f"Không gian tìm kiếm (Search Space Size): {len(distractor_ids)} keyframes")

    # Encode Text Queries
    with torch.no_grad():
        txt_array = encode_texts_raw(valid_queries, batch_size=64) 
        txt_feats = torch.from_numpy(txt_array).float().to(DEVICE)
        if text_head is not None:
            txt_feats = text_head(txt_feats)
        txt_matrix_norm = F.normalize(txt_feats, dim=-1).cpu().numpy()

    # 4. Tính toán Ma trận Độ tương đồng & Metrics (N_queries x N_distractors)
    sims = txt_matrix_norm @ dist_matrix_norm.T
    ranked_indices = np.argsort(-sims, axis=1)

    print("\n================ BÁO CÁO ĐÁNH GIÁ THỰC TẾ ================")
    for k in top_k_list:
        hits = 0
        for i, target_idx in enumerate(gt_indices):
            if target_idx in ranked_indices[i, :k]:
                hits += 1
        recall_k = (hits / len(valid_queries)) * 100
        print(f"Realistic Recall@{k:2d}: {recall_k:.2f}% ({hits}/{len(valid_queries)})")

    ranks = []
    for i, target_idx in enumerate(gt_indices):
        rank = int(np.where(ranked_indices[i] == target_idx)[0][0]) + 1
        ranks.append(1.0 / rank)
    mrr = float(np.mean(ranks))
    print(f"Realistic MRR       : {mrr:.4f}")
    print("==========================================================")


def main():
    pairs = build_pairs(use_caption_first=True)
    _, val_pairs = train_val_split(pairs)
    evaluate_with_distractors(val_pairs)


if __name__ == "__main__":
    main()