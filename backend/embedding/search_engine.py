"""Search Engine tích hợp FAISS Local Index, Single Query, Temporal Search và Auto-Translation."""
import json
import logging
import re
from pathlib import Path
import numpy as np
import torch
import faiss
from PIL import Image
import open_clip
from deep_translator import GoogleTranslator

from backend.config import (
    METADATA_PATH,
    BTC_CLIP_FEATURES_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    DEVICE,
    ENSEMBLE_EMBED_DIM
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """Bộ máy tìm kiếm Vector FAISS đa năng (Single & Temporal Search)."""
    def __init__(self, device=DEVICE):
        self.device = device
        self.index = None
        self.metadata = []
        self.model_clip = None
        self.tokenizer = None

    def build_index(self, limit: int = 0):
        """Build FAISS local index từ các file .npy."""
        out_dir = BTC_CLIP_FEATURES_DIR / "ensemble_features"
        if not out_dir.exists():
            logger.error("Chưa có thư mục features. Vui lòng trích xuất đặc trưng trước.")
            return None, None

        items = []
        with open(METADATA_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit > 0 and i >= limit: break
                if line.strip(): items.append(json.loads(line))

        vectors = []
        metadata = []
        dim = 0

        for idx, item in enumerate(items):
            key = f"{item['video_id']}_{item.get('frame_id', 0)}"
            feat_path = out_dir / f"{key}.npy"

            if feat_path.exists():
                vec = np.load(feat_path)
                if dim == 0: dim = vec.shape[0]
                vectors.append(vec)
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
            return None, None

        vectors_np = np.vstack(vectors).astype(np.float32)
        faiss.normalize_L2(vectors_np)

        index = faiss.IndexFlatIP(dim)
        index.add(vectors_np)

        faiss.write_index(index, str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.index = index
        self.metadata = metadata
        logger.info("Đã xây dựng xong FAISS Index (%dd) với %d keyframes.", dim, len(metadata))
        return index, metadata

    def load_index(self):
        """Nạp FAISS index từ ổ cứng."""
        if FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(FAISS_METADATA_PATH, encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info("Đã nạp thành công FAISS Index (%dd, %d items).", self.index.d, len(self.metadata))
            return True
        return False

    def _ensure_text_encoder(self):
        if self.model_clip is None:
            self.model_clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=self.device)
            self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

    def search_single(self, query_text: str, top_k: int = 10):
        """Truy vấn đơn lẻ (Single Query) với tự động dịch Vi -> En."""
        if self.index is None:
            if not self.load_index():
                raise RuntimeError("Chưa có FAISS Index. Vui lòng build index trước.")

        self._ensure_text_encoder()

        # Dịch Vi -> En
        try:
            query_en = GoogleTranslator(source='vi', target='en').translate(query_text)
        except Exception:
            query_en = query_text

        text_tokens = self.tokenizer([query_en]).to(self.device)
        with torch.no_grad():
            text_feat = self.model_clip.encode_text(text_tokens)
            text_feat = (text_feat / text_feat.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze(0)

        target_dim = self.index.d
        if target_dim > text_feat.shape[0]:
            padding = np.zeros(target_dim - text_feat.shape[0], dtype=np.float32)
            query_super = np.concatenate([text_feat, padding]).astype(np.float32)
        else:
            query_super = text_feat.astype(np.float32)

        query_super = query_super.reshape(1, -1)
        faiss.normalize_L2(query_super)

        scores, indices = self.index.search(query_super, top_k)
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            score = float(scores[0][i])
            if 0 <= idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = score
                # Dịch video_title từ En sang Vi cho hiển thị
                try:
                    item["video_title_vi"] = GoogleTranslator(source='en', target='vi').translate(item.get("video_title", ""))
                except:
                    item["video_title_vi"] = item.get("video_title", "")
                results.append(item)

        return {"query_vi": query_text, "query_en": query_en, "results": results}

    def search_temporal(self, temporal_query_text: str, max_gap_sec: float = 120.0, top_k_candidates: int = 50, top_k: int = 5):
        """Tìm kiếm theo thời gian đa giai đoạn (Multi-stage Temporal Search)."""
        raw_sub_queries = re.split(r'->|sau đó|then|rồi', temporal_query_text, flags=re.IGNORECASE)
        sub_queries = [sq.strip() for sq in raw_sub_queries if sq.strip()]

        if len(sub_queries) < 2:
            return self.search_single(temporal_query_text, top_k=top_k)

        sub_candidates = []
        for sq in sub_queries:
            res = self.search_single(sq, top_k=top_k_candidates)
            sub_candidates.append(res["results"])

        candidates_A, candidates_B = sub_candidates[0], sub_candidates[1]

        video_to_B = {}
        for item_b in candidates_B:
            vid = item_b["video_id"]
            if vid not in video_to_B: video_to_B[vid] = []
            video_to_B[vid].append(item_b)

        temporal_matches = []
        for item_a in candidates_A:
            vid = item_a["video_id"]
            pts_a = item_a.get("pts_time", 0.0)
            score_a = item_a["score"]

            if vid in video_to_B:
                for item_b in video_to_B[vid]:
                    pts_b = item_b.get("pts_time", 0.0)
                    score_b = item_b["score"]

                    if 0 < (pts_b - pts_a) <= max_gap_sec:
                        time_gap = pts_b - pts_a
                        combined_score = score_a + score_b - (time_gap / max_gap_sec) * 0.05
                        temporal_matches.append({
                            "video_id": vid,
                            "event_a": item_a,
                            "event_b": item_b,
                            "time_gap": time_gap,
                            "combined_score": combined_score
                        })

        temporal_matches.sort(key=lambda x: x["combined_score"], reverse=True)
        return {
            "query": temporal_query_text,
            "sub_queries": sub_queries,
            "max_gap_sec": max_gap_sec,
            "matches": temporal_matches[:top_k]
        }
