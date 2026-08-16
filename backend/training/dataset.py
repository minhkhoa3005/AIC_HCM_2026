"""
Dataset, Samplers & Helper functions cho huấn luyện Projection Head và Temporal Video Encoder.
"""
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset

from backend.config import BTC_CLIP_FEATURES_DIR, METADATA_PATH, VAL_SPLIT_RATIO, resolve_path
from backend.embedding.clip_encoder import encode_text_raw, encode_texts_raw

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        return vec
    return vec / norm


# ---------------------------------------------------------------------------
# Auto-discover batch folder cache (shared with build_index_features.py logic)
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

    return _batch_folder_cache


from functools import lru_cache

@lru_cache(maxsize=1024)
def _cached_np_load(path_str: str) -> np.ndarray:
    return np.load(path_str)


def _load_keyframe_feature(video_id: str, frame_stem: str, ordinal: int | None = None) -> np.ndarray | None:
    batch_guess = video_id.split("_")[0]
    cache = _get_batch_folders()

    # 1. Try stacked per-video .npy
    stacked_candidates = [
        BTC_CLIP_FEATURES_DIR / batch_guess / f"{video_id}.npy",
        BTC_CLIP_FEATURES_DIR / f"{video_id}.npy",
    ]
    for parent_dir in cache.get(video_id, []):
        discovered = parent_dir / f"{video_id}.npy"
        if discovered not in stacked_candidates:
            stacked_candidates.append(discovered)

    for stacked_path in stacked_candidates:
        if stacked_path.exists():
            try:
                arr = _cached_np_load(str(stacked_path))
                if ordinal is not None and ordinal < len(arr):
                    return _normalize(arr[ordinal].reshape(-1))
            except Exception:
                pass

    # 2. Try per-frame .npy
    frame_candidates = [
        BTC_CLIP_FEATURES_DIR / batch_guess / video_id / f"{frame_stem}.npy",
        BTC_CLIP_FEATURES_DIR / video_id / f"{frame_stem}.npy",
    ]
    if frame_stem.isdigit():
        n = int(frame_stem)
        for fmt in (f"{n:06d}", f"{n:04d}", f"{n:05d}"):
            if fmt != frame_stem:
                frame_candidates.append(BTC_CLIP_FEATURES_DIR / batch_guess / video_id / f"{fmt}.npy")

    for parent_dir in cache.get(video_id, []):
        per_frame_dir = parent_dir / video_id
        if per_frame_dir.is_dir():
            frame_candidates.append(per_frame_dir / f"{frame_stem}.npy")
            if frame_stem.isdigit():
                frame_candidates.append(per_frame_dir / f"{int(frame_stem):06d}.npy")

    for path in frame_candidates:
        if path.exists():
            try:
                arr = _cached_np_load(str(path))
                return _normalize(arr.reshape(-1))
            except Exception:
                pass
    return None


def build_pairs(use_caption_first: bool = True) -> List[Dict]:
    """Tạo danh sách training pairs từ metadata.jsonl và các file vector feature CLIP."""
    if not METADATA_PATH.exists():
        logger.error("Không tìm thấy metadata tại %s. Hãy chạy import_btc_data.py trước.", METADATA_PATH)
        return []

    items = []
    with open(METADATA_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(json.loads(line))

    raw_candidates = []
    unique_texts = set()

    for item in items:
        video_id = item["video_id"]
        frame_stem = resolve_path(item.get("path", "")).stem
        ordinal = item.get("keyframe_ordinal")

        img_emb = _load_keyframe_feature(video_id, frame_stem, ordinal)
        if img_emb is None:
            continue

        # Chọn văn bản làm label
        if use_caption_first and item.get("caption") and len(item["caption"].strip()) >= 5:
            text = item["caption"].strip()
        else:
            text = (item.get("text") or item.get("object_tags") or item.get("video_title") or "").strip()

        if not text:
            continue

        raw_candidates.append((video_id, str(resolve_path(item.get("path", ""))), text, img_emb))
        unique_texts.add(text)

    # Mã hóa theo batch tất cả các chuỗi text duy nhất (tăng tốc độ gấp 50 lần)
    text_list = list(unique_texts)
    encoded_vecs = encode_texts_raw(text_list, batch_size=128)
    text_cache = {t: vec.astype("float32") for t, vec in zip(text_list, encoded_vecs)}

    pairs = []
    for video_id, feat_path, text, img_emb in raw_candidates:
        pairs.append(
            {
                "video_id": video_id,
                "feat_path": feat_path,
                "train_text": text,
                "img_emb": img_emb.astype("float32"),
                "txt_emb": text_cache[text],
            }
        )

    logger.info("Đã tạo %d training pairs từ %d metadata items.", len(pairs), len(items))
    return pairs


def build_sequence_pairs() -> List[Dict]:
    """Gom nhóm keyframes theo video để tạo dữ liệu chuỗi (sequence) cho Temporal Video Encoder."""
    pairs = build_pairs(use_caption_first=True)
    by_video: Dict[str, List[Dict]] = defaultdict(list)
    for p in pairs:
        by_video[p["video_id"]].append(p)

    seq_pairs = []
    for v_id, v_items in by_video.items():
        if len(v_items) < 2:
            continue
        v_items = sorted(v_items, key=lambda x: x.get("frame_id", x.get("keyframe_ordinal", 0)))
        seq_matrix = np.stack([it["img_emb"] for it in v_items]).astype("float32")
        txt_emb = np.mean([it["txt_emb"] for it in v_items], axis=0).astype("float32")
        norm = np.linalg.norm(txt_emb)
        if norm > 1e-8:
            txt_emb = txt_emb / norm
        seq_pairs.append(
            {
                "video_id": v_id,
                "seq_tensor": seq_matrix,
                "txt_emb": txt_emb,
            }
        )

    logger.info("Đã tạo %d chuỗi video (sequences) cho Temporal Training.", len(seq_pairs))
    return seq_pairs


def train_val_split(pairs: List[Dict], val_ratio: float = VAL_SPLIT_RATIO, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Tách tập train/val theo video_id để tránh rò rỉ thông tin giữa 2 tập."""
    video_ids = sorted(list({p["video_id"] for p in pairs}))
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    n_val = max(1, int(len(video_ids) * val_ratio))
    val_vids = set(video_ids[:n_val])

    train_pairs = [p for p in pairs if p["video_id"] not in val_vids]
    val_pairs = [p for p in pairs if p["video_id"] in val_vids]
    return train_pairs, val_pairs


class CachedEmbeddingDataset(Dataset):
    """Dataset trả về (img_emb, txt_emb) dạng Tensor cho Projection Head."""

    def __init__(self, pairs: List[Dict]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.pairs[idx]
        img_t = torch.from_numpy(item["img_emb"])
        txt_t = torch.from_numpy(item["txt_emb"])
        return img_t, txt_t


class TemporalVideoDataset(Dataset):
    """Dataset trả về (seq_tensor, txt_emb) cho Temporal Video Encoder."""

    def __init__(self, seq_pairs: List[Dict]):
        self.seq_pairs = seq_pairs

    def __len__(self) -> int:
        return len(self.seq_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.seq_pairs[idx]
        seq_t = torch.from_numpy(item["seq_tensor"])
        txt_t = torch.from_numpy(item["txt_emb"])
        return seq_t, txt_t


def temporal_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences có độ dài khác nhau về cùng max_len trong batch.

    Tránh crash khi DataLoader cố stack tensors có shape khác nhau (mỗi video
    có số keyframes khác nhau).
    """
    seqs, txts = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    dim = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, dim)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0], :] = s
    return padded, torch.stack(txts)


class UniqueVideoBatchSampler(BatchSampler):
    """BatchSampler đảm bảo không có 2 mẫu nào trùng video_id trong cùng 1 batch."""

    def __init__(self, pairs: List[Dict], batch_size: int, drop_last: bool = True):
        self.pairs = pairs
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.video_to_indices = defaultdict(list)
        for idx, p in enumerate(pairs):
            self.video_to_indices[p["video_id"]].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        available_indices = {
            v_id: list(idxs) for v_id, idxs in self.video_to_indices.items()
        }

        while len(available_indices) >= self.batch_size:
            selected_videos = random.sample(
                list(available_indices.keys()), self.batch_size
            )
            batch = []

            for v_id in selected_videos:
                idx = available_indices[v_id].pop(
                    random.randrange(len(available_indices[v_id]))
                )
                batch.append(idx)

                if not available_indices[v_id]:
                    del available_indices[v_id]

            yield batch

    def __len__(self) -> int:
        n_videos = len(self.video_to_indices)
        return n_videos // self.batch_size