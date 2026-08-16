"""Push BTC CLIP feature vectors lên Remote Vector Database (Qdrant).

Luồng chính (canonical path cho AIC):
  1. Đọc metadata.jsonl (đã import qua import_btc_data.py)
  2. Load BTC CLIP features (.npy) — raw 512d vector, KHÔNG re-encode từ video
  3. Push thẳng lên Qdrant kèm metadata payload

Vector space đảm bảo nhất quán:
  - Database vectors: raw CLIP features (512d) từ BTC
  - Search queries: raw CLIP text encoding (512d) qua encode_text_raw()
  → Cùng vector space, cosine similarity chính xác.

Nếu cần dùng Projection Head (fine-tuned), dùng flag --projected:
  - Database vectors: raw → image_head → 256d
  - Search queries: raw → text_head → 256d  (clip_encoder.encode_text() tự apply)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import (  # noqa: E402
    BTC_CLIP_FEATURES_DIR,
    DEVICE,
    EMBED_DIM,
    METADATA_PATH,
    PROJECTED_DIM,
    PROJECTION_HEAD_PATH,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_PREFER_GRPC,
    QDRANT_URL,
    USE_REMOTE_VECTOR_DB,
    VECTOR_DB_TYPE,
    resolve_path,
)
from backend.embedding.remote_index import (  # noqa: E402
    init_remote_collection,
    upsert_vectors_remote,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-discover batch folder cache (reused from build_index_features.py)
# ---------------------------------------------------------------------------
_batch_folder_cache: dict[str, list[Path]] | None = None


def _get_batch_folders() -> dict[str, list[Path]]:
    """Scan BTC_CLIP_FEATURES_DIR once and build mapping: video_id -> [parent_dirs]."""
    global _batch_folder_cache
    if _batch_folder_cache is not None:
        return _batch_folder_cache

    _batch_folder_cache = {}
    if not BTC_CLIP_FEATURES_DIR.exists():
        return _batch_folder_cache

    for npy_file in BTC_CLIP_FEATURES_DIR.rglob("*.npy"):
        vid_guess = npy_file.stem
        parent = npy_file.parent
        _batch_folder_cache.setdefault(vid_guess, [])
        if parent not in _batch_folder_cache[vid_guess]:
            _batch_folder_cache[vid_guess].append(parent)

    for entry in BTC_CLIP_FEATURES_DIR.rglob("*"):
        if entry.is_dir():
            vid_guess = entry.name
            _batch_folder_cache.setdefault(vid_guess, [])
            if entry.parent not in _batch_folder_cache[vid_guess]:
                _batch_folder_cache[vid_guess].append(entry.parent)

    logger.info("Auto-discovered %d video entries in CLIP features directory.", len(_batch_folder_cache))
    return _batch_folder_cache


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        return vec
    return vec / norm


def _video_feature_candidates(video_id: str) -> list[Path]:
    """Tìm file .npy stacked per-video."""
    batch_guess = video_id.split("_")[0]
    candidates = [
        BTC_CLIP_FEATURES_DIR / batch_guess / f"{video_id}.npy",
        BTC_CLIP_FEATURES_DIR / f"{video_id}.npy",
    ]
    cache = _get_batch_folders()
    for parent_dir in cache.get(video_id, []):
        discovered = parent_dir / f"{video_id}.npy"
        if discovered not in candidates:
            candidates.append(discovered)
    return candidates


def _feature_candidates(video_id: str, frame_name: str) -> list[Path]:
    """Tìm file .npy per-frame."""
    batch_guess = video_id.split("_")[0]
    candidates = [
        BTC_CLIP_FEATURES_DIR / batch_guess / video_id / f"{frame_name}.npy",
        BTC_CLIP_FEATURES_DIR / video_id / f"{frame_name}.npy",
    ]
    if frame_name.isdigit():
        n = int(frame_name)
        for fmt in (f"{n:06d}", f"{n:04d}", f"{n:05d}"):
            if fmt != frame_name:
                candidates.append(BTC_CLIP_FEATURES_DIR / batch_guess / video_id / f"{fmt}.npy")
    cache = _get_batch_folders()
    for parent_dir in cache.get(video_id, []):
        per_frame_dir = parent_dir / video_id
        if per_frame_dir.is_dir():
            candidates.append(per_frame_dir / f"{frame_name}.npy")
            if frame_name.isdigit():
                candidates.append(per_frame_dir / f"{int(frame_name):06d}.npy")
    return candidates


def _load_features_for_remote(items: list[dict]) -> tuple[list[dict], np.ndarray]:
    """Load BTC CLIP features cho từng keyframe metadata, trả về items kèm vectors.

    Logic giống build_index_features.py: ưu tiên stacked .npy, fallback per-frame .npy.
    """
    by_video: dict[str, list[dict]] = {}
    for item in items:
        by_video.setdefault(item["video_id"], []).append(item)

    kept_items: list[dict] = []
    vectors: list[np.ndarray] = []
    skipped = 0

    for video_id, video_items in tqdm(by_video.items(), desc="Loading BTC CLIP features"):
        video_items = sorted(video_items, key=lambda x: x.get("keyframe_ordinal", 0))

        # Try stacked .npy per video
        stacked = None
        for path in _video_feature_candidates(video_id):
            if path.exists():
                try:
                    stacked = np.load(str(path))
                    break
                except Exception as exc:
                    logger.error("[%s] Lỗi khi nạp %s: %s", video_id, path, exc)

        for item in video_items:
            vec = None
            ordinal = item.get("keyframe_ordinal")

            # Priority 1: stacked .npy
            if stacked is not None and ordinal is not None and ordinal < len(stacked):
                vec = _normalize(np.asarray(stacked[ordinal]).reshape(-1))

            # Priority 2: per-frame .npy
            if vec is None:
                frame_name = resolve_path(item.get("path", "")).stem
                for path in _feature_candidates(video_id, frame_name):
                    if path.exists():
                        try:
                            arr = np.load(str(path))
                            vec = _normalize(arr.reshape(-1))
                            break
                        except Exception:
                            pass

            if vec is None:
                skipped += 1
                continue

            kept_items.append(item)
            vectors.append(vec.astype("float32"))

    if not vectors:
        return [], np.empty((0, 0), dtype="float32")

    matrix = np.stack(vectors).astype("float32")
    logger.info(
        "Loaded %d feature vectors (dim=%d), bỏ qua %d keyframe thiếu feature.",
        len(vectors), matrix.shape[1], skipped,
    )
    return kept_items, matrix



def main():
    parser = argparse.ArgumentParser(description="Push BTC CLIP features to Remote Vector DB (Qdrant)")
    parser.add_argument("--collection", type=str, default=QDRANT_COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection from scratch")

    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for upsert")
    args = parser.parse_args()

    # 1. Load metadata
    if not METADATA_PATH.exists():
        logger.error("Không tìm thấy %s. Hãy chạy import_btc_data.py trước.", METADATA_PATH)
        return

    with open(METADATA_PATH, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if not items:
        logger.error("Metadata rỗng.")
        return

    logger.info("Loaded %d metadata items từ %s", len(items), METADATA_PATH)

    # 2. Load BTC CLIP features
    kept_items, matrix = _load_features_for_remote(items)
    if not kept_items:
        logger.error("Không load được feature vector nào. Kiểm tra %s.", BTC_CLIP_FEATURES_DIR)
        return

    # 3. Sử dụng raw CLIP features
    logger.info("Push raw CLIP features (dim=%d)", matrix.shape[1])

    # 4. Prepare payload cho Qdrant
    enriched_items = []
    for item in kept_items:
        payload = {
            "video_id": item.get("video_id", ""),
            "clip_id": item.get("clip_id", ""),
            "video_title": item.get("video_title", ""),
            "frame_id": int(item.get("frame_id", 0)),
            "pts_time": float(item.get("pts_time", 0.0)),
            "fps": float(item.get("fps", 25.0)),
            "keyframe_ordinal": int(item.get("keyframe_ordinal", 0)),
            "path": item.get("path", ""),
            "text": item.get("text", ""),
            "caption": item.get("caption", ""),
            "object_tags": item.get("object_tags", ""),
            "scene_id": item.get("scene_id", ""),
        }
        enriched_items.append(payload)

    # 5. Push to Remote
    vector_dim = matrix.shape[1]
    logger.info("Pushing %d vectors (dim=%d) to Qdrant collection '%s'...", len(enriched_items), vector_dim, args.collection)

    if args.recreate:
        init_remote_collection(collection_name=args.collection, vector_size=vector_dim, recreate=True)

    count = upsert_vectors_remote(
        items=enriched_items,
        vectors=matrix,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )

    logger.info("THÀNH CÔNG: Đã push %d vectors lên Qdrant collection '%s'.", count, args.collection)
    logger.info("ℹ Vector space: RAW CLIP (%dd). Search query dùng encode_text_raw().", vector_dim)


if __name__ == "__main__":
    main()
