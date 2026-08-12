import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import VIDEOS_DIR, KEYFRAMES_DIR, INDEX_DIR, METADATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCENE_THRESHOLD = 30.0  # Threshold for cv2.absdiff mean difference. Giảm nếu muốn cắt nhiều ảnh hơn, tăng nếu muốn ít ảnh hơn.
PROCESS_FPS = 5         # Tốc độ khung hình xử lý ảo (chỉ xử lý 5 hình/giây để tiết kiệm tính toán).
DOWNSCALE_DIM = (128, 72) # Cực nhỏ để thuật toán absdiff chạy siêu nhanh.


def process_video(video_path: Path) -> List[Dict[str, Any]]:
    video_id = video_path.stem
    output_dir = KEYFRAMES_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    # Tính toán hệ số bỏ qua khung hình (skip factor) để đạt tốc độ PROCESS_FPS
    # vd: fps=30, PROCESS_FPS=5 => step=6 (cứ 6 frame đọc 1 lần)
    step = max(1, int(round(fps / PROCESS_FPS)))
    
    prev_gray = None
    frame_idx = 0
    keyframe_ordinal = 0
    metadata_rows = []
    
    # Bắt buộc lưu khung hình đầu tiên của mỗi video
    force_save_next = True

    while True:
        # Tối ưu hóa: Dùng grab() để lướt qua frame nhanh chóng mà không decode RGB
        if frame_idx % step != 0 and not force_save_next:
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
            continue
            
        ret, frame = cap.read()
        if not ret:
            break
            
        # Thu nhỏ hình ảnh và chuyển sang xám (Grayscale) để so sánh tốc độ cao
        small = cv2.resize(frame, DOWNSCALE_DIM, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        is_scene_change = False
        if force_save_next or prev_gray is None:
            is_scene_change = True
            force_save_next = False
        else:
            # Thuật toán tìm độ lệch tuyệt đối
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = np.mean(diff)
            if mean_diff > SCENE_THRESHOLD:
                is_scene_change = True
                
        if is_scene_change:
            # Khi nhận diện cắt cảnh -> Lưu khung hình GỐC (chất lượng cao)
            pts_time = frame_idx / fps
            frame_name = f"{frame_idx:06d}"
            save_path = output_dir / f"{frame_name}.jpg"
            
            # Lưu ảnh nén JPEG chất lượng 95
            cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Ghi nhận Metadata
            metadata_rows.append({
                "video_id": video_id,
                "frame_id": frame_name,
                "pts_time": pts_time,
                "keyframe_ordinal": keyframe_ordinal,
                "clip_id": f"{video_id}_{frame_name}",
                "path": str(save_path.relative_to(ROOT_DIR)).replace("\\", "/")
            })
            keyframe_ordinal += 1
            prev_gray = gray
            
        frame_idx += 1
        
    cap.release()
    logger.info(f"Processed {video_id}: Extracted {len(metadata_rows)} keyframes.")
    return metadata_rows

def main():
    if not VIDEOS_DIR.exists():
        logger.error(f"Thư mục chứa video {VIDEOS_DIR} không tồn tại.")
        return
        
    video_files = list(VIDEOS_DIR.glob("*.mp4"))
    if not video_files:
        logger.warning(f"Không tìm thấy file .mp4 nào trong {VIDEOS_DIR}.")
        return
        
    logger.info(f"Tìm thấy {len(video_files)} videos. Bắt đầu trích xuất...")
    
    all_metadata = []
    
    # Multiprocessing: Xử lý nhiều video cùng một lúc bằng tất cả lõi CPU
    max_workers = max(1, os.cpu_count() - 1)
    logger.info(f"Chạy đa luồng với {max_workers} CPU cores.")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(process_video, vf): vf for vf in video_files}
        
        for future in concurrent.futures.as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                rows = future.result()
                all_metadata.extend(rows)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý {video_path.name}: {e}")
                
    # Sắp xếp metadata chuẩn xác
    all_metadata.sort(key=lambda x: (x["video_id"], x["keyframe_ordinal"]))
    
    # Ghi đè file metadata.jsonl
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for row in all_metadata:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    logger.info(f"Hoàn thành! Đã trích xuất tổng cộng {len(all_metadata)} keyframes.")
    logger.info(f"Toàn bộ Metadata đã được lưu tại: {METADATA_PATH}")

if __name__ == "__main__":
    main()
