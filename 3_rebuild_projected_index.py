import json
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from backend.config import (
    METADATA_PATH,
    BTC_CLIP_FEATURES_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    DEVICE,
    PROJECTION_HEAD_PATH
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Re-define the Projection Head class
class AICProjectionHead(nn.Module):
    def __init__(self, input_dim=2304, output_dim=768, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        out = self.net(x)
        out = F.normalize(out, p=2, dim=-1)
        return out

def main():
    logger.info("=== BƯỚC 3: XÂY DỰNG LẠI FAISS INDEX VỚI PROJECTION HEAD ===")
    
    out_dir = BTC_CLIP_FEATURES_DIR / "ensemble_features"
    if not out_dir.exists():
        logger.error("Chưa có features, vui lòng chạy bước 1 trước.")
        return
        
    if not PROJECTION_HEAD_PATH.exists() and not Path("projection_head_latest.pth").exists():
        logger.error("Chưa có mô hình Projection Head. Vui lòng chạy bước 2 trước.")
        return

    items = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    vectors = []
    metadata = []
    dim = 0
    model = None
    
    for idx, item in enumerate(items):
        key = f"{item['video_id']}_{item.get('frame_id', 0)}"
        feat_path = out_dir / f"{key}.npy"
        
        if feat_path.exists():
            vec = np.load(feat_path)
            if dim == 0:
                dim = vec.shape[0]
                logger.info("Phát hiện số chiều của Raw Vector: %d", dim)
                
                # Load model now that we know the dimension
                model = AICProjectionHead(input_dim=dim).to(DEVICE)
                ckpt_path = PROJECTION_HEAD_PATH if PROJECTION_HEAD_PATH.exists() else "projection_head_latest.pth"
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                logger.info("Đã nạp Projection Head thành công.")
                
            # Project vector
            vec_tensor = torch.tensor(vec).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                proj_vec = model(vec_tensor).cpu().numpy().squeeze(0)
                
            vectors.append(proj_vec)
            payload = {
                "id": idx,
                "video_id": item["video_id"],
                "video_title": item.get("video_title", "Chưa rõ"),
                "frame_id": item.get("frame_id", 0),
                "pts_time": item.get("pts_time", 0.0),
                "path": item.get("path", "")
            }
            metadata.append(payload)

    if not vectors:
        logger.warning("Không tìm thấy vector nào để đưa vào FAISS.")
        return

    # Build FAISS Index
    vectors_np = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(vectors_np)
    
    proj_dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(proj_dim)
    index.add(vectors_np)
    
    # Save back to same paths
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    logger.info("Đã xây dựng xong FAISS Index (Projected %d chiều) với %d keyframes.", proj_dim, len(metadata))

if __name__ == "__main__":
    main()
