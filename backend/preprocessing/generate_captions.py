"""
generate_captions.py
Trích xuất mô tả hình ảnh (Visual Captioning) từ Keyframe dùng BLIP-2.
Kết quả được lưu vào trường 'caption' trong file metadata JSONL — TÁCH BIỆT
khỏi trường 'text' (transcript lời nói), vì đây là 2 nguồn dữ liệu train khác
nhau: 'caption' mô tả NỘI DUNG HÌNH ẢNH (khớp với văn phong đề thi AIC),
còn 'text' là lời thoại nói ra trong video (chỉ phù hợp cho dạng Q&A).
"""
import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from backend.config import DEVICE, INDEX_DIR, METADATA_PATH, VIDEO_METADATA_DIR, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _flush_to_disk(items: list[dict]) -> None:
    """Ghi metadata tổng hợp + đồng bộ lại file theo từng video."""
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_video: dict[str, list[dict]] = {}
    for item in items:
        by_video.setdefault(item["video_id"], []).append(item)
    for video_id, video_items in by_video.items():
        with open(VIDEO_METADATA_DIR / f"{video_id}.jsonl", "w", encoding="utf-8") as f:
            for item in video_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_keyframe_captions(batch_size: int = 16, save_every: int = 10, limit: int = 0) -> None:
    """Quét metadata.jsonl, sinh caption cho các keyframe chưa có trường 'caption'.

    Args:
        batch_size: số ảnh xử lý cùng lúc trong 1 lần inference BLIP-2.
        save_every: sau mỗi bao nhiêu batch thì flush kết quả ra đĩa (chống
            mất tiến độ nếu crash/OOM giữa chừng — BLIP-2 khá nặng).
        limit: giới hạn số keyframe xử lý, để test nhanh trước khi chạy full
            dataset (0 = không giới hạn).
    """
    if not METADATA_PATH.exists():
        logger.error("Không tìm thấy %s. Hãy chạy import_btc_data trước.", METADATA_PATH)
        return

    # ── Đọc và chuẩn bị metadata TRƯỚC khi nạp model ──────────────────────
    # Mục đích: tránh chiếm RAM/VRAM nặng (BLIP-2 ~5-10GB) nếu không có
    # keyframe nào cần sinh caption.
    items = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    if limit > 0:
        items = items[:limit]
        logger.info("Giới hạn xử lý %d keyframe đầu tiên (chế độ test).", limit)

    # Đếm số keyframe thực sự cần xử lý (chưa có caption hoặc caption quá ngắn)
    pending = [
        item for item in items
        if not (item.get("caption") and len(item["caption"].strip()) >= 5)
    ]
    if not pending:
        logger.info("Tất cả %d keyframe đã có caption hợp lệ. Không cần chạy BLIP-2.", len(items))
        return

    logger.info(
        "Cần sinh caption cho %d/%d keyframe. Đang nạp mô hình BLIP-2...",
        len(pending), len(items),
    )

    # ── Nạp model CHỈ KHI có việc cần làm ─────────────────────────────────
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()

    updated_count = 0
    batches_since_save = 0

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        images_to_process = []
        indices_to_update = []

        for idx, item in enumerate(batch):
            # Nếu đã có caption hợp lệ thì bỏ qua (cơ chế cache)
            if item.get("caption") and len(item["caption"].strip()) >= 5:
                continue

            img_path = resolve_path(item.get("path", ""))
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    images_to_process.append(img)
                    indices_to_update.append(idx)
                except Exception as e:
                    logger.warning("Không thể đọc ảnh %s: %s", img_path, e)

        if not images_to_process:
            continue

        # Inference batch với Prompt định hướng
        CAPTION_PROMPT = (
            "Describe this image in detail, focusing on: "
            "people and their clothing colors, actions being performed, "
            "the location or setting, and notable objects visible."
        )
        prompts = [CAPTION_PROMPT] * len(images_to_process)
        inputs = processor(
            images=images_to_process,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=4,
                length_penalty=1.2,
                repetition_penalty=1.3,
            )
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for batch_idx, caption_text in zip(indices_to_update, generated_texts):
            clean_caption = caption_text.strip()
            batch[batch_idx]["caption"] = clean_caption
            updated_count += 1

        batches_since_save += 1
        logger.info("Đã sinh caption cho %d/%d keyframe...", i + len(batch), len(items))

        # Flush định kỳ ra đĩa để không mất tiến độ nếu crash giữa chừng
        if batches_since_save >= save_every:
            _flush_to_disk(items)
            batches_since_save = 0
            logger.info("Đã lưu checkpoint tạm thời.")

    # Ghi lần cuối + đồng bộ lại xuống từng file theo video
    _flush_to_disk(items)

    logger.info("Hoàn tất! Đã cập nhật caption cho %d keyframe vào %s.", updated_count, METADATA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh caption hình ảnh cho keyframe bằng BLIP-2")
    parser.add_argument("--batch-size", type=int, default=16, help="Số ảnh xử lý cùng lúc mỗi lần inference")
    parser.add_argument("--save-every", type=int, default=10, help="Flush ra đĩa sau mỗi N batch")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số keyframe để test nhanh (0=all)")
    args = parser.parse_args()
    generate_keyframe_captions(batch_size=args.batch_size, save_every=args.save_every, limit=args.limit)