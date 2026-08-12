"""Script tạo dữ liệu mẫu (Sample Demo Data) để test luồng Preprocessing, Training & Push Remote DB.

Sinh dữ liệu theo đúng schema keyframe-level hiện tại (giống output của
scripts/import_btc_data.py) — KHÔNG dùng schema clip 20s cũ nữa, để demo phản
ánh đúng luồng thật (keyframe_ordinal, pts_time, frame_source, scene_id...).

Lưu ý: đây chỉ là dữ liệu giả lập để test code chạy được, "path" trỏ tới file
ảnh không tồn tại thật — các bước cần đọc ảnh thật (build_index_features.py
đọc .npy, không đọc ảnh trực tiếp) vẫn chạy được, nhưng bước nào cần mở file
ảnh thật sẽ cần thay bằng dữ liệu thật.
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import INDEX_DIR, KEYFRAMES_DIR, METADATA_PATH, VIDEO_METADATA_DIR

SAMPLE_TEXTS = [
    "Diễn giả mặc áo đỏ đang phát biểu tại cuộc họp báo ngoài trời phía sau có nhiều cây xanh.",
    "Người phụ nữ mặc váy đỏ đang cầm ly màu xanh tại bữa tiệc âm nhạc.",
    "Vận động viên thực hiện cú nhảy cao chạy đà giậm nhảy bay qua xà và tiếp đất.",
    "Cảnh một người đang mở laptop làm việc trong phòng khách rộng rãi.",
    "Nhóm thanh niên đang chơi bóng rổ trên sân ngoài trời vào buổi chiều.",
    "Ca sĩ đang biểu diễn trên sân khấu lớn với dàn đèn chiếu sáng rực rỡ.",
    "Đoạn clip giới thiệu món ăn đường phố Việt Nam với phở và bánh mì.",
    "Cảnh quay flycam từ trên cao xuống bờ biển xanh cát trắng nắng vàng.",
    "Ô tô màu đen đang di chuyển trên đường cao tốc hướng về thành phố.",
    "Robot tự động đang thao tác lắp ráp linh kiện trong nhà máy thông minh.",
]

SAMPLE_OBJECTS = [
    "Person, Red shirt, Tree, Microphone",
    "Woman, Red dress, Glass, Party",
    "Athlete, High jump, Bar, Mat",
    "Person, Laptop, Table, Chair",
    "Basketball, Court, Player",
    "Singer, Stage, Light, Speaker",
    "Food, Noodle, Bread, Bowl",
    "Sea, Beach, Sand, Coast",
    "Car, Highway, Road",
    "Robot, Factory, Machine",
]


def create_sample_dataset(n_videos: int = 5, keyframes_per_video: int = 5, fps: float = 25.0):
    """Tạo metadata.jsonl giả lập theo schema keyframe-level để test toàn bộ pipeline."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    all_items = []

    for v_idx in range(1, n_videos + 1):
        video_id = f"L{v_idx:02d}_V{v_idx:03d}"
        video_items = []

        # Giả lập các keyframe cách nhau 4 giây (giống mật độ keyframe thực tế của BTC)
        for k_idx in range(keyframes_per_video):
            pts_time = k_idx * 4.0
            frame_id = int(round(pts_time * fps))
            frame_stem = f"{k_idx:04d}"
            frame_path = KEYFRAMES_DIR / video_id / f"{frame_stem}.jpg"

            sample_idx = (v_idx * keyframes_per_video + k_idx) % len(SAMPLE_TEXTS)
            text = SAMPLE_TEXTS[sample_idx]
            obj_tags = SAMPLE_OBJECTS[sample_idx]

            item = {
                "video_id": video_id,
                "video_title": f"Video Demo {video_id}",
                "clip_id": f"kf_{frame_stem}",
                "path": str(frame_path),
                "original_video_path": "",
                "start": round(max(0.0, pts_time - 1.0), 3),
                "end": round(pts_time + 1.0, 3),
                "start_frame": frame_id,
                "end_frame": frame_id,
                "keyframe_ordinal": k_idx,
                "frame_id": frame_id,
                "fps": fps,
                "pts_time": round(pts_time, 6),
                "frame_source": "demo_sample",
                "text": text,
                "object_tags": obj_tags,
                "scene_id": "",
                "scene_rank": 0,
            }
            video_items.append(item)
            all_items.append(item)

        with open(VIDEO_METADATA_DIR / f"{video_id}.jsonl", "w", encoding="utf-8") as f:
            for item in video_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Success] Created {len(all_items)} sample keyframes in {METADATA_PATH}")
    print(
        "Note: This script generates dummy metadata without corresponding .npy CLIP vectors. "
        "Use this sample dataset to test metadata imports and schema pipelines."
    )


if __name__ == "__main__":
    create_sample_dataset()