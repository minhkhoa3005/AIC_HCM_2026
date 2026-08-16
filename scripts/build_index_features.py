"""Build two-level FAISS indexes from organizer CLIP feature files.

Outputs:
- data/index/video.index: keyframe-level index for exact frame localization.
- data/index/scene.index: scene-level index for coarse retrieval.
- data/index/index_metadata.json: keyframe metadata enriched with scene_id
  (only keyframes that have a real CLIP vector — has_feature=True).
- data/index/scene_metadata.json: scene metadata and keyframe membership.
- data/index/video_metadata/<video_id>.jsonl: FULL per-video metadata,
  including keyframes that had no CLIP feature available (has_feature=False)
  — kept for traceability, but excluded from the FAISS indexes/scenes.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import (  # noqa: E402
    BTC_CLIP_FEATURES_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    METADATA_PATH,
    QDRANT_COLLECTION_NAME,
    SCENE_FAISS_INDEX_PATH,
    SCENE_METADATA_PATH,
    SMART_CUT_MAX_SCENE_KEYFRAMES,
    SMART_CUT_MIN_SCENE_KEYFRAMES,
    SMART_CUT_SIMILARITY_THRESHOLD,
    USE_REMOTE_VECTOR_DB,
    VIDEO_METADATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        return vec
    return vec / norm


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        logger.error("%s not found. Run scripts/import_btc_data.py first.", path)
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON at line %d: %s", line_num, exc)
    return rows


# ---------------------------------------------------------------------------
# Auto-discover batch folder cache
# ---------------------------------------------------------------------------
# BTC may name batch folders inconsistently across years/batches, e.g.:
#   L01, Keyframes_L01, CLIP_L01, Batch_1, Features_L01, ...
# Instead of guessing from video_id.split("_")[0], we scan the actual
# directory tree once and cache the results.

_batch_folder_cache: dict[str, list[Path]] | None = None


def _get_batch_folders() -> dict[str, list[Path]]:
    """Scan BTC_CLIP_FEATURES_DIR once and build a mapping: video_id -> [parent_dirs].

    Also indexes batch-prefix guesses so both exact and heuristic lookups work.
    """
    global _batch_folder_cache
    if _batch_folder_cache is not None:
        return _batch_folder_cache

    _batch_folder_cache = {}
    if not BTC_CLIP_FEATURES_DIR.exists():
        return _batch_folder_cache

    # Index stacked .npy files: clip-features/<batch>/L01_V001.npy
    for npy_file in BTC_CLIP_FEATURES_DIR.rglob("*.npy"):
        video_id_guess = npy_file.stem
        parent = npy_file.parent
        _batch_folder_cache.setdefault(video_id_guess, [])
        if parent not in _batch_folder_cache[video_id_guess]:
            _batch_folder_cache[video_id_guess].append(parent)

    # Index per-frame directories: clip-features/<batch>/L01_V001/<frame>.npy
    for entry in BTC_CLIP_FEATURES_DIR.rglob("*"):
        if entry.is_dir():
            video_id_guess = entry.name
            _batch_folder_cache.setdefault(video_id_guess, [])
            if entry.parent not in _batch_folder_cache[video_id_guess]:
                _batch_folder_cache[video_id_guess].append(entry.parent)

    logger.info(
        "Auto-discovered %d video entries in CLIP features directory.",
        len(_batch_folder_cache),
    )
    return _batch_folder_cache


def _video_feature_candidates(video_id: str) -> list[Path]:
    """Return candidate paths for a stacked per-video .npy file."""
    batch_guess = video_id.split("_")[0]
    candidates = [
        # Heuristic guesses (fast path)
        BTC_CLIP_FEATURES_DIR / batch_guess / f"{video_id}.npy",
        BTC_CLIP_FEATURES_DIR / f"{video_id}.npy",
    ]

    # Auto-discovered paths from actual directory scan
    cache = _get_batch_folders()
    for parent_dir in cache.get(video_id, []):
        discovered = parent_dir / f"{video_id}.npy"
        if discovered not in candidates:
            candidates.append(discovered)

    return candidates


def _feature_candidates(video_id: str, frame_name: str) -> list[Path]:
    """Return candidate paths for a per-frame .npy file."""
    batch_guess = video_id.split("_")[0]
    candidates = [
        BTC_CLIP_FEATURES_DIR / batch_guess / video_id / f"{frame_name}.npy",
        BTC_CLIP_FEATURES_DIR / video_id / f"{frame_name}.npy",
    ]

    # Try zero-padded variants if frame_name is numeric
    if frame_name.isdigit():
        n = int(frame_name)
        for fmt in (f"{n:06d}", f"{n:04d}", f"{n:05d}"):
            if fmt != frame_name:
                candidates.append(BTC_CLIP_FEATURES_DIR / batch_guess / video_id / f"{fmt}.npy")

    # Auto-discovered parent directories
    cache = _get_batch_folders()
    for parent_dir in cache.get(video_id, []):
        per_frame_dir = parent_dir / video_id
        if per_frame_dir.is_dir():
            candidates.append(per_frame_dir / f"{frame_name}.npy")
            if frame_name.isdigit():
                candidates.append(per_frame_dir / f"{int(frame_name):06d}.npy")

    return candidates


def _load_individual_feature(item: dict) -> np.ndarray | None:
    frame_name = resolve_path(item.get("path", "")).stem
    for path in _feature_candidates(item["video_id"], frame_name):
        if not path.exists():
            continue
        arr = np.load(str(path))
        return _normalize(arr.reshape(-1))
    return None


def load_features_by_video(
    items: list[dict], limit: int = 0
) -> tuple[list[dict], list[dict], np.ndarray]:
    """Attach organizer CLIP features to metadata rows.

    Returns:
        kept_items: chỉ chứa item CÓ vector thật — dùng để build FAISS/scene.
        all_metadata: chứa TẤT CẢ item, kể cả thiếu feature — dùng để ghi
            JSONL đầy đủ (has_feature=False cho item bị loại).
        vectors matrix: khớp 1-1 theo thứ tự với kept_items.
    """
    by_video: dict[str, list[dict]] = {}
    for item in items:
        by_video.setdefault(item["video_id"], []).append(item)

    kept_items: list[dict] = []
    all_metadata: list[dict] = []
    vectors: list[np.ndarray] = []
    skipped = 0
    missing_ordinal = 0

    for video_id, video_items in tqdm(by_video.items(), desc="Loading organizer CLIP features"):
        if any("keyframe_ordinal" not in item for item in video_items):
            missing_ordinal += 1
            logger.warning(
                "[%s] metadata missing 'keyframe_ordinal' — re-run import_btc_data.py "
                "to regenerate metadata, otherwise vectors may be mismatched.",
                video_id,
            )

        # Dùng keyframe_ordinal (thứ tự scan gốc, KHÔNG phải frame_id) để khớp .npy —
        # vì file .npy của BTC được xếp theo thứ tự scan keyframe gốc.
        video_items = sorted(video_items, key=lambda x: x.get("keyframe_ordinal", 0))
        if limit and len(kept_items) >= limit:
            break

        stacked = None
        for path in _video_feature_candidates(video_id):
            if path.exists():
                try:
                    stacked = np.load(str(path))
                    break
                except Exception as exc:
                    logger.error("[%s] Lỗi khi nạp %s: %s", video_id, path, exc)

        num_expected = len(video_items)
        if stacked is not None and len(stacked) != num_expected:
            logger.warning(
                "[%s] LỆCH SỐ LƯỢNG: .npy có %d vector, metadata có %d keyframe.",
                video_id, len(stacked), num_expected,
            )

        for item in video_items:
            if limit and len(kept_items) >= limit:
                break

            vec = None
            ordinal = item.get("keyframe_ordinal")
            if stacked is not None and ordinal is not None and ordinal < len(stacked):
                vec = _normalize(np.asarray(stacked[ordinal]).reshape(-1))
            if vec is None:
                try:
                    vec = _load_individual_feature(item)
                except Exception as exc:
                    logger.debug("Could not load feature for %s/%s: %s", video_id, item.get("clip_id"), exc)

            item = dict(item)  # tránh sửa item gốc
            if vec is None:
                # KHÔNG fabricate zero-vector — loại khỏi vector/scene pipeline,
                # nhưng vẫn giữ trong metadata đầy đủ để không mất thông tin keyframe này.
                skipped += 1
                item["has_feature"] = False
                all_metadata.append(item)
                logger.error(
                    "[%s] Missing feature vector for keyframe ordinal %s - skipped from index.",
                    video_id, ordinal,
                )
                continue

            item["has_feature"] = True
            kept_items.append(item)
            all_metadata.append(item)
            vectors.append(vec.astype("float32"))

    if not vectors:
        return [], all_metadata, np.empty((0, 0), dtype="float32")

    matrix = np.stack(vectors).astype("float32")
    logger.info(
        "Loaded %d feature vectors (dim=%d), loại %d keyframe thiếu feature khỏi index.",
        len(vectors), matrix.shape[1], skipped,
    )
    if missing_ordinal:
        logger.warning(
            "%d videos had metadata rows missing 'keyframe_ordinal'. "
            "Vector-to-frame_id mapping for those videos may be unreliable.",
            missing_ordinal,
        )
    return kept_items, all_metadata, matrix


def _new_scene(video_id: str, scene_index: int, members: list[int], items: list[dict], vectors: np.ndarray) -> dict:
    scene_id = f"{video_id}_S{scene_index:04d}"
    scene_items = [items[i] for i in members]
    scene_vec = _normalize(vectors[members].mean(axis=0))
    return {
        "scene_id": scene_id,
        "video_id": video_id,
        "video_title": scene_items[0].get("video_title", video_id),
        "start": min(float(item.get("start", item.get("pts_time", 0.0))) for item in scene_items),
        "end": max(float(item.get("end", item.get("pts_time", 0.0))) for item in scene_items),
        "start_frame": min(int(item.get("frame_id", 0)) for item in scene_items),
        "end_frame": max(int(item.get("frame_id", 0)) for item in scene_items),
        "frame_id": int(scene_items[len(scene_items) // 2].get("frame_id", 0)),
        "clip_id": scene_id,
        "text": "",
        "object_tags": scene_items[0].get("object_tags", ""),
        "keyframe_indices": members,
        "_vector": scene_vec.astype("float32"),
    }


def smart_cut_scenes(items: list[dict], vectors: np.ndarray) -> tuple[list[dict], np.ndarray, list[dict]]:
    """Group adjacent keyframes into scenes using CLIP cosine similarity drops."""
    by_video: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        by_video.setdefault(item["video_id"], []).append(idx)

    scene_rows = []
    scene_vectors = []
    enriched = [dict(item) for item in items]

    for video_id, indices in by_video.items():
        indices = sorted(indices, key=lambda i: (items[i].get("frame_id", 0), items[i].get("clip_id", "")))
        scene_index = 0
        current = [indices[0]]
        previous = indices[0]

        for idx in indices[1:]:
            similarity = float(np.dot(vectors[previous], vectors[idx]))
            too_different = similarity < SMART_CUT_SIMILARITY_THRESHOLD
            too_long = len(current) >= SMART_CUT_MAX_SCENE_KEYFRAMES
            can_cut = len(current) >= SMART_CUT_MIN_SCENE_KEYFRAMES
            if (too_different and can_cut) or too_long:
                scene = _new_scene(video_id, scene_index, current, items, vectors)
                scene_rows.append(scene)
                scene_vectors.append(scene.pop("_vector"))
                scene_index += 1
                current = []
            current.append(idx)
            previous = idx

        if current:
            scene = _new_scene(video_id, scene_index, current, items, vectors)
            scene_rows.append(scene)
            scene_vectors.append(scene.pop("_vector"))

    for scene in scene_rows:
        member_count = len(scene["keyframe_indices"])
        for rank, item_idx in enumerate(scene["keyframe_indices"]):
            enriched[item_idx]["scene_id"] = scene["scene_id"]
            enriched[item_idx]["scene_rank"] = rank
            enriched[item_idx]["scene_size"] = member_count
            enriched[item_idx]["scene_start_frame"] = scene["start_frame"]
            enriched[item_idx]["scene_end_frame"] = scene["end_frame"]

    scene_matrix = np.stack(scene_vectors).astype("float32") if scene_vectors else np.empty((0, 0), dtype="float32")
    logger.info("Smart cutting produced %d scenes from %d keyframes.", len(scene_rows), len(items))
    return enriched, scene_matrix, scene_rows


def _write_json(path: Path, rows: list[dict]) -> None:
    serializable = []
    for row in rows:
        clean = dict(row)
        clean["keyframe_indices"] = [int(i) for i in clean.get("keyframe_indices", [])]
        serializable.append(clean)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False, indent=2)


def _write_jsonl_by_video(rows: list[dict]) -> None:
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(row["video_id"], []).append(row)
    for video_id, video_rows in by_video.items():
        with open(VIDEO_METADATA_DIR / f"{video_id}.jsonl", "w", encoding="utf-8") as fh:
            for row in video_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene-level and keyframe-level indexes from BTC features")
    parser.add_argument("--remote", action="store_true", help="Push keyframe vectors to remote vector DB")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of keyframes for testing (0=all)")
    args = parser.parse_args()

    items = _load_jsonl(METADATA_PATH)
    if not items:
        return

    kept_items, all_metadata, keyframe_matrix = load_features_by_video(items, limit=args.limit)
    if not kept_items:
        logger.error("No CLIP features loaded. Check %s.", BTC_CLIP_FEATURES_DIR)
        return

    # smart_cut_scenes / FAISS chỉ dùng kept_items (item có vector thật)
    enriched_items, scene_matrix, scene_rows = smart_cut_scenes(kept_items, keyframe_matrix)

    # Ghép lại: item nào đã qua smart_cut_scenes thì lấy bản enriched (có scene_id...),
    # item nào bị loại (has_feature=False) thì giữ nguyên bản gốc — không mất dữ liệu.
    enriched_by_key = {(it["video_id"], it["clip_id"]): it for it in enriched_items}
    final_metadata = []
    for item in all_metadata:
        key = (item["video_id"], item["clip_id"])
        final_metadata.append(enriched_by_key.get(key, item))

    import faiss

    keyframe_index = faiss.IndexFlatIP(keyframe_matrix.shape[1])
    keyframe_index.add(keyframe_matrix)
    faiss.write_index(keyframe_index, str(FAISS_INDEX_PATH))

    if scene_matrix.shape[0] == 0 or scene_matrix.ndim < 2 or scene_matrix.shape[1] == 0:
        logger.warning(
            "Không có scene vector nào để index (scene_matrix rỗng). "
            "Bỏ qua việc tạo scene.index — kiểm tra SMART_CUT_MIN_SCENE_KEYFRAMES."
        )
        scene_index = None
    else:
        scene_index = faiss.IndexFlatIP(scene_matrix.shape[1])
        scene_index.add(scene_matrix)
        faiss.write_index(scene_index, str(SCENE_FAISS_INDEX_PATH))

    _write_json(FAISS_METADATA_PATH, enriched_items)   # chỉ item có vector — khớp đúng với FAISS index
    _write_json(SCENE_METADATA_PATH, scene_rows)
    _write_jsonl_by_video(final_metadata)               # đầy đủ, kể cả has_feature=False

    logger.info("Keyframe index: %s (%d vectors)", FAISS_INDEX_PATH, keyframe_index.ntotal)
    if scene_index is not None:
        logger.info("Scene index: %s (%d vectors)", SCENE_FAISS_INDEX_PATH, scene_index.ntotal)
    else:
        logger.info("Scene index: skipped (no scene vectors)")
    logger.info("Keyframe metadata: %s", FAISS_METADATA_PATH)
    logger.info("Scene metadata: %s", SCENE_METADATA_PATH)

    if args.remote or USE_REMOTE_VECTOR_DB:
        logger.info("Pushing keyframe vectors to remote vector database...")
        try:
            from backend.embedding.remote_index import init_remote_collection, upsert_vectors_remote

            init_remote_collection(collection_name=QDRANT_COLLECTION_NAME, vector_size=keyframe_matrix.shape[1])
            count = upsert_vectors_remote(
                items=enriched_items,
                vectors=keyframe_matrix,
                collection_name=QDRANT_COLLECTION_NAME,
            )
            logger.info("Pushed %d keyframe vectors to %s.", count, QDRANT_COLLECTION_NAME)
        except Exception as exc:
            logger.warning("Could not push to remote vector DB: %s", exc)


if __name__ == "__main__":
    main()