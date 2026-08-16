"""Import organizer BTC keyframes and metadata into keyframe-level JSONL.

Each row in metadata.jsonl is one organizer keyframe. frame_id is resolved with
the following priority:
  1. Organizer map-keyframes file (BTC_MAP_KEYFRAMES_DIR), if it lists this
     keyframe explicitly — most trustworthy, always preferred.
  2. Exact PTS lookup via ffprobe on the source video (handles VFR correctly),
     if the source video is available in VIDEOS_DIR.
  3. Linear fallback (ordinal / fps), only when neither of the above is
     available.

Each row also keeps 'keyframe_ordinal' (the scan order within its video
folder) — this MUST be used, not frame_id, to index into the organizer's
per-video CLIP .npy feature files, because those files are stacked in scan
order, not frame_id order.
"""
import argparse
import bisect
import csv
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import (  # noqa: E402
    BTC_MAP_KEYFRAMES_DIR,
    BTC_MEDIA_INFO_DIR,
    INDEX_DIR,
    KEYFRAMES_DIR,
    METADATA_PATH,
    VIDEO_METADATA_DIR,
    VIDEOS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _row_get(row: dict, *names: str) -> Any:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for name in names:
        key = name.lower()
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def _video_id_from_path(path: Path, base_dir: Path) -> str:
    rel = path.relative_to(base_dir)
    if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
        return path.stem
    return rel.parts[-1]


def _key_aliases(frame_name: str, ordinal: int | None = None) -> set[str]:
    aliases = {frame_name, Path(frame_name).stem}
    stem = Path(frame_name).stem
    try:
        n = int(stem)
        aliases.update({str(n), f"{n:04d}", f"{n:05d}", f"{n:06d}"})
    except ValueError:
        pass
    if ordinal is not None:
        aliases.update({str(ordinal), f"{ordinal:04d}", f"{ordinal:05d}", f"{ordinal:06d}"})
    return aliases


def _find_video_file(video_id: str) -> Path | None:
    """Tìm file video gốc tương ứng với video_id trong VIDEOS_DIR (nếu có)."""
    for ext in (".mp4", ".mov", ".mkv"):
        candidate = VIDEOS_DIR / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    matches = list(VIDEOS_DIR.rglob(f"{video_id}.*"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# VFR-aware exact frame lookup via ffprobe packet PTS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _get_video_pts_list(video_path: str) -> tuple:
    """Đọc toàn bộ timestamp (PTS) thực tế của video bằng ffprobe.

    Dùng để mapping chính xác pts_time -> frame_id cho video VFR (Variable
    Frame Rate), thay vì suy diễn tuyến tính (pts_time * fps) vốn có thể lệch
    dần với video dài / VFR.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        pts_list = sorted(float(p.strip()) for p in out.stdout.strip().split("\n") if p.strip())
        return tuple(pts_list)
    except Exception as exc:
        logger.debug("Không đọc được PTS từ %s bằng ffprobe: %s", video_path, exc)
        return tuple()


def _lookup_exact_frame_by_pts(pts_time: float, video_path: str | None, fallback_fps: float) -> tuple[int, str]:
    """Tìm frame_id chính xác nhất dựa trên PTS thực tế của video.

    Trả về (frame_id, source_tag). Nếu đọc được PTS thực qua ffprobe, dùng
    binary search để tìm frame gần pts_time nhất. Nếu không, fallback về suy
    diễn tuyến tính.
    """
    if video_path and Path(video_path).exists():
        pts_list = _get_video_pts_list(str(video_path))
        if pts_list:
            idx = bisect.bisect_left(pts_list, pts_time)
            if idx == 0:
                return 0, "ffprobe_pts"
            if idx == len(pts_list):
                return len(pts_list) - 1, "ffprobe_pts"
            before, after = pts_list[idx - 1], pts_list[idx]
            chosen = idx if (after - pts_time) < (pts_time - before) else (idx - 1)
            return chosen, "ffprobe_pts"

    fallback_frame_id = int(round(pts_time * fallback_fps)) if fallback_fps > 0 else 0
    return fallback_frame_id, "linear_fallback"


# ---------------------------------------------------------------------------
# Media info (fps / duration / title) — advisory only
# ---------------------------------------------------------------------------

def load_media_info() -> dict[str, dict]:
    """Load fps/duration metadata. This is advisory; frame_id comes from maps."""
    info: dict[str, dict] = {}
    if not BTC_MEDIA_INFO_DIR.exists():
        logger.warning("Media-info dir not found: %s", BTC_MEDIA_INFO_DIR)
        return info

    for file_path in BTC_MEDIA_INFO_DIR.rglob("*"):
        if file_path.suffix.lower() not in {".csv", ".json"}:
            continue
        try:
            if file_path.suffix.lower() == ".csv":
                with open(file_path, encoding="utf-8-sig") as fh:
                    rows = list(csv.DictReader(fh))
            else:
                with open(file_path, encoding="utf-8-sig") as fh:
                    data = json.load(fh)
                rows = data if isinstance(data, list) else [data]

            fallback_video_id = _video_id_from_path(file_path, BTC_MEDIA_INFO_DIR)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                video_id = str(_row_get(row, "video_id", "video", "id") or fallback_video_id).strip()
                if not video_id:
                    continue
                info[video_id] = {
                    "fps": _as_float(_row_get(row, "fps", "FPS", "frame_rate"), 25.0),
                    "duration": _as_float(_row_get(row, "duration", "length"), 0.0),
                    "title": _row_get(row, "title", "name") or video_id.replace("_", " "),
                }
        except Exception as exc:
            logger.debug("Could not parse media-info file %s: %s", file_path, exc)

    logger.info("Loaded media-info for %d videos.", len(info))
    return info


# ---------------------------------------------------------------------------
# Organizer keyframe maps (frame_id / pts_time as given by BTC, when present)
# ---------------------------------------------------------------------------

def _record_keyframe(mapping: dict, video_id: str, frame_key: Any, record: dict) -> None:
    if not video_id or frame_key in (None, ""):
        return
    bucket = mapping.setdefault(video_id, {})
    for alias in _key_aliases(str(frame_key)):
        bucket[alias] = record


def load_map_keyframes() -> dict[str, dict[str, dict]]:
    """Load organizer keyframe maps.

    Output: mapping[video_id][frame_name_alias] -> {frame_id, pts_time, fps}.
    The parser is intentionally permissive because AIC map files vary by batch.
    """
    mapping: dict[str, dict[str, dict]] = {}
    if not BTC_MAP_KEYFRAMES_DIR.exists():
        logger.warning("Map-keyframes dir not found: %s", BTC_MAP_KEYFRAMES_DIR)
        return mapping

    for file_path in BTC_MAP_KEYFRAMES_DIR.rglob("*"):
        if file_path.suffix.lower() not in {".csv", ".json"}:
            continue
        try:
            if file_path.suffix.lower() == ".csv":
                with open(file_path, encoding="utf-8-sig") as fh:
                    rows = list(csv.DictReader(fh))
            else:
                with open(file_path, encoding="utf-8-sig") as fh:
                    data = json.load(fh)
                rows = data if isinstance(data, list) else data.get("keyframes", [data])

            fallback_video_id = _video_id_from_path(file_path, BTC_MAP_KEYFRAMES_DIR)
            for ordinal, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                video_id = str(_row_get(row, "video_id", "video", "id") or fallback_video_id).strip()
                key = _row_get(
                    row,
                    "frame_name",
                    "filename",
                    "file_name",
                    "image",
                    "keyframe",
                    "keyframe_id",
                    "n",
                )
                fps = _as_float(_row_get(row, "fps", "FPS", "frame_rate"), 25.0)
                pts_time = _as_float(_row_get(row, "pts_time", "timestamp", "time", "sec", "seconds"), 0.0)
                explicit_frame = _row_get(row, "frame_id", "frame_idx", "frame_index", "frame", "n")
                frame_id = _as_int(explicit_frame, _as_int(round(pts_time * fps), 0))
                record = {
                    "frame_id": frame_id,
                    "pts_time": pts_time if pts_time > 0 else (frame_id / fps if fps > 0 else 0.0),
                    "fps": fps,
                    "source": "organizer_map",
                }
                _record_keyframe(mapping, video_id, key if key is not None else ordinal, record)
                _record_keyframe(mapping, video_id, ordinal, record)
        except Exception as exc:
            logger.debug("Could not parse map-keyframes file %s: %s", file_path, exc)

    logger.info("Loaded keyframe maps for %d videos.", len(mapping))
    return mapping




# ---------------------------------------------------------------------------
# Keyframe scanning
# ---------------------------------------------------------------------------

def scan_keyframes() -> dict[str, list[Path]]:
    """Trả về {video_id: [frame_path...]} theo đúng thứ tự scan gốc.

    Thứ tự này (chỉ số vị trí trong list) chính là 'keyframe_ordinal' — PHẢI
    khớp với thứ tự vector trong file .npy CLIP features của BTC (BTC xếp
    theo thứ tự scan keyframe, không phải theo frame_id).
    """
    videos: dict[str, list[Path]] = {}
    if not KEYFRAMES_DIR.exists():
        logger.warning("Keyframes dir not found: %s", KEYFRAMES_DIR)
        return videos

    for frame_path in sorted(KEYFRAMES_DIR.rglob("*")):
        if frame_path.is_file() and frame_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            videos.setdefault(frame_path.parent.name, []).append(frame_path)

    logger.info(
        "Scanned keyframes for %d videos, total %d frames.",
        len(videos),
        sum(len(paths) for paths in videos.values()),
    )
    return videos


# ---------------------------------------------------------------------------
# Frame lookup — priority: organizer map > exact ffprobe PTS > linear fallback
# ---------------------------------------------------------------------------

def _lookup_frame(
    frame_map: dict[str, dict],
    frame_path: Path,
    ordinal: int,
    fps: float,
    video_path: str | None = None,
) -> dict:
    """Xác định frame_id/pts_time cho 1 keyframe, theo thứ tự ưu tiên:

    1. Organizer map (BTC cung cấp sẵn) — đáng tin cậy nhất, giữ nguyên.
    2. Tra PTS chính xác qua ffprobe trên video gốc (xử lý đúng VFR), nếu có
       video gốc trong VIDEOS_DIR.
    3. Suy diễn tuyến tính (ordinal / fps) — chỉ khi không có 2 cái trên.
    """
    # Ưu tiên 1: organizer map
    for alias in _key_aliases(frame_path.stem, ordinal):
        if alias in frame_map:
            return frame_map[alias]

    # Ưu tiên 2: PTS chính xác qua ffprobe (nếu có video gốc)
    pts_time_guess = ordinal / fps if fps > 0 else 0.0
    if video_path:
        exact_frame_id, source_tag = _lookup_exact_frame_by_pts(pts_time_guess, video_path, fps)
        return {
            "frame_id": exact_frame_id,
            "pts_time": exact_frame_id / fps if fps > 0 else 0.0,
            "fps": fps,
            "source": source_tag,
        }

    # Ưu tiên 3: fallback dựa theo tên file / ordinal
    fallback_frame_id = _as_int(frame_path.stem, ordinal)
    return {
        "frame_id": fallback_frame_id,
        "pts_time": fallback_frame_id / fps if fps > 0 else 0.0,
        "fps": fps,
        "source": "filename_fallback",
    }


# ---------------------------------------------------------------------------
# Dynamic transcript assignment by PTS window
# ---------------------------------------------------------------------------

def assign_text_by_pts(video_items: list[dict], segments: list[dict]) -> None:
    """Gán transcript cho từng keyframe dựa trên window thời gian động quanh pts_time.

    Window được tính tự động dựa trên khoảng cách tới keyframe lân cận để tránh:
    - Bỏ sót transcript lời nói gần đó khi keyframe thưa.
    - Chồng lấn transcript giữa các keyframe quá gần nhau.
    """
    if not video_items or not segments:
        return

    sorted_items = sorted(video_items, key=lambda x: x.get("pts_time", 0.0))
    n_items = len(sorted_items)

    for i, item in enumerate(sorted_items):
        pts = item["pts_time"]

        prev_pts = sorted_items[i - 1]["pts_time"] if i > 0 else None
        next_pts = sorted_items[i + 1]["pts_time"] if i < n_items - 1 else None

        gaps = []
        if prev_pts is not None:
            gaps.append(abs(pts - prev_pts))
        if next_pts is not None:
            gaps.append(abs(next_pts - pts))

        if gaps:
            half_gap = min(gaps) / 2.0
            window = max(0.5, min(2.0, half_gap))
        else:
            window = 1.0

        overlapping = [
            seg["text"].strip()
            for seg in segments
            if seg.get("text") and seg["start"] <= pts + window and seg["end"] >= pts - window
        ]

        unique_texts = list(dict.fromkeys(overlapping))
        item["text"] = " ".join(unique_texts).strip()


# ---------------------------------------------------------------------------
# Cache / Idempotency helpers
# ---------------------------------------------------------------------------

def _is_video_metadata_usable(video_id: str, min_rows: int = 1) -> bool:
    """Kiểm tra xem file JSONL metadata của video đã tồn tại và có nội dung hợp lệ chưa."""
    jsonl_path = VIDEO_METADATA_DIR / f"{video_id}.jsonl"
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return False
    try:
        count = 0
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
                    if count >= min_rows:
                        return True
    except Exception:
        return False
    return False


def _load_existing_video_metadata(video_id: str) -> list[dict]:
    """Đọc lại metadata đã được xử lý trước đó từ file .jsonl có sẵn."""
    jsonl_path = VIDEO_METADATA_DIR / f"{video_id}.jsonl"
    rows = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_btc_metadata(
    limit_videos: int = 0,
    with_transcript: bool = False,
    force: bool = False,
) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    keyframes = scan_keyframes()
    if not keyframes:
        logger.error("No keyframe data found. Expected: %s/<batch>/<video_id>/*.jpg", KEYFRAMES_DIR)
        return

    video_ids = sorted(keyframes)
    if limit_videos > 0:
        video_ids = video_ids[:limit_videos]
        logger.info("Limiting to first %d videos.", limit_videos)

    media_info = None
    keyframe_maps = None

    all_items = []
    skipped_count = 0

    for video_id in video_ids:
        if not force and _is_video_metadata_usable(video_id):
            logger.info("[%s] Metadata JSONL đã tồn tại hợp lệ. Skipping (dùng --force để ghi đè).", video_id)
            existing_items = _load_existing_video_metadata(video_id)
            all_items.extend(existing_items)
            skipped_count += 1
            continue

        if media_info is None:
            media_info = load_media_info()
            keyframe_maps = load_map_keyframes()

        info = media_info.get(video_id, {})
        fps = _as_float(info.get("fps"), 25.0)
        duration = _as_float(info.get("duration"), 0.0)
        raw_title = str(info.get("title") or video_id.replace("_", " "))
        
        # Dịch title sang Tiếng Anh
        from backend.embedding.clip_encoder import translate_vi_to_en
        title = translate_vi_to_en(raw_title)
        
        frame_map = keyframe_maps.get(video_id, {})

        video_path_obj = _find_video_file(video_id)
        video_path_str = str(video_path_obj) if video_path_obj else None

        video_items = []

        for ordinal, frame_path in enumerate(keyframes[video_id]):
            frame_info = _lookup_frame(frame_map, frame_path, ordinal, fps, video_path=video_path_str)
            frame_id = int(frame_info["frame_id"])
            pts_time = float(frame_info["pts_time"])
            item_fps = float(frame_info.get("fps") or fps)
            item = {
                "video_id": video_id,
                "video_title": title,
                "clip_id": f"kf_{frame_path.stem}",
                "path": str(frame_path.relative_to(ROOT_DIR)).replace("\\", "/") if frame_path.is_relative_to(ROOT_DIR) else str(frame_path),
                "original_video_path": str(Path(video_path_str).relative_to(ROOT_DIR)).replace("\\", "/") if video_path_str and Path(video_path_str).is_relative_to(ROOT_DIR) else (video_path_str or ""),
                "start": round(max(0.0, pts_time - 1.0), 3),
                "end": round(min(duration, pts_time + 1.0) if duration > 0 else pts_time + 1.0, 3),
                "start_frame": frame_id,
                "end_frame": frame_id,
                "keyframe_ordinal": ordinal,
                "frame_id": frame_id,
                "fps": round(item_fps, 3),
                "pts_time": round(pts_time, 6),
                "frame_source": frame_info.get("source", "unknown"),
                "text": "",
                "object_tags": "",
                "scene_id": "",
                "scene_rank": 0,
            }
            video_items.append(item)

        if with_transcript and video_path_str:
            try:
                from backend.preprocessing.transcribe import transcribe_video

                segments = transcribe_video(Path(video_path_str))
                assign_text_by_pts(video_items, segments)
            except Exception as exc:
                logger.warning("Could not transcribe %s: %s", video_id, exc)
        elif with_transcript:
            logger.debug("[%s] no source video found for transcription, skipping.", video_id)

        with open(VIDEO_METADATA_DIR / f"{video_id}.jsonl", "w", encoding="utf-8") as fh:
            for item in video_items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

        all_items.extend(video_items)
        mapped = sum(1 for item in video_items if item["frame_source"] == "organizer_map")
        exact_pts = sum(1 for item in video_items if item["frame_source"] == "ffprobe_pts")
        logger.info(
            "[%s] Processed %d keyframes (%d organizer-mapped, %d exact-pts, %d fallback)",
            video_id,
            len(video_items),
            mapped,
            exact_pts,
            len(video_items) - mapped - exact_pts,
        )

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        for item in all_items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        "Done. Wrote %d total keyframe rows to %s (Skipped %d existing videos).",
        len(all_items),
        METADATA_PATH,
        skipped_count,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Entrypoint: Import BTC AIC keyframes & metadata")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos to import (0=all)")
    parser.add_argument(
        "--with-transcript",
        action="store_true",
        help="Transcribe source videos using Whisper and assign text per keyframe",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing metadata JSONL files even if already generated",
    )
    args = parser.parse_args()
    build_btc_metadata(
        limit_videos=args.limit,
        with_transcript=args.with_transcript,
        force=args.force,
    )