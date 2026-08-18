"""Generate visual captions for keyframes and write them back to metadata.

The caption field is used as the primary text signal for LoRA CLIP training
when transcripts are unavailable. Captions are intentionally visual: people,
clothing, actions, setting, objects, and visible text.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from backend.config import CURRENT_METADATA_PATH, DEVICE, VIDEO_METADATA_DIR, resolve_path

METADATA_PATH = CURRENT_METADATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


PROMPTS = {
    "aic": (
        "Write one factual English caption for visual retrieval of this video frame. "
        "Describe only visible evidence. Include the main people or objects, their actions, "
        "clothing colors, object colors, spatial relations, location, vehicles, signs, logos, "
        "and readable text when visible. Preserve numbers and counts. Do not guess identity, "
        "intent, hidden details, or the answer to a question. Use one detailed but concise sentence."
    ),
    "dense": (
        "Describe the image in detail for image retrieval. Include visible people, clothing, "
        "actions, objects, background, location, colors, and any text in the image."
    ),
    "short": "A concise description of the image:",
}


def _read_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        logger.error("%s not found. Run scripts/import_btc_data.py first.", METADATA_PATH)
        return []

    rows = []
    with open(METADATA_PATH, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid metadata line %d: %s", line_num, exc)
    return rows


def _flush_to_disk(items: list[dict]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    VIDEO_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_video: dict[str, list[dict]] = {}
    for item in items:
        by_video.setdefault(item["video_id"], []).append(item)

    for video_id, video_items in by_video.items():
        with open(VIDEO_METADATA_DIR / f"{video_id}.jsonl", "w", encoding="utf-8") as fh:
            for item in video_items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def _caption_is_usable(item: dict, min_words: int) -> bool:
    caption = str(item.get("caption") or "").strip()
    return len(caption.split()) >= min_words


def _item_time(item: dict, fallback: int) -> float:
    """Read the VFR-safe timestamp used to group keyframes into windows."""
    for field in ("pts_time", "timestamp", "start", "frame_time"):
        value = item.get(field)
        try:
            if value is not None and str(value).strip() != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    for field in ("frame_id", "keyframe_ordinal"):
        value = item.get(field)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return float(fallback)


def _caption_windows(
    items: list[dict],
    *,
    window_seconds: float,
    force: bool,
    min_words: int,
    limit: int,
) -> tuple[list[int], dict[tuple[str, int], list[int]], dict[tuple[str, int], int]]:
    """Build one representative job per video/time window."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    windows: dict[tuple[str, int], list[int]] = {}
    times: dict[int, float] = {}
    for idx, item in enumerate(items):
        path = resolve_path(item.get("path", ""))
        if not path.exists():
            continue
        timestamp = _item_time(item, idx)
        key = (str(item.get("video_id", "unknown")), int(timestamp // window_seconds))
        windows.setdefault(key, []).append(idx)
        times[idx] = timestamp

    representatives: dict[tuple[str, int], int] = {}
    for key, members in windows.items():
        window_start = key[1] * window_seconds
        window_center = window_start + window_seconds / 2.0
        representatives[key] = min(
            members,
            key=lambda idx: (abs(times[idx] - window_center), times[idx], idx),
        )

    ordered_keys = sorted(
        windows,
        key=lambda key: (key[0], key[1]),
    )
    if limit > 0:
        ordered_keys = ordered_keys[:limit]
    selected_windows = {key: windows[key] for key in ordered_keys}
    pending = [
        representatives[key]
        for key in ordered_keys
        if force or not _caption_is_usable(items[representatives[key]], min_words)
    ]
    return pending, selected_windows, {key: representatives[key] for key in ordered_keys}


def _propagate_window_captions(
    items: list[dict],
    windows: dict[tuple[str, int], list[int]],
    representatives: dict[tuple[str, int], int],
    *,
    model_name: str,
    model_type: str,
    prompt_name: str,
    window_seconds: float,
    force: bool,
) -> int:
    """Copy a representative caption to keyframes in the same time window."""
    updated = 0
    for key, members in windows.items():
        representative = items[representatives[key]]
        caption = str(representative.get("caption") or "").strip()
        if not caption:
            continue
        for idx in members:
            if not force and _caption_is_usable(items[idx], min_words=1):
                continue
            items[idx]["caption"] = caption
            items[idx]["caption_model"] = model_name
            items[idx]["caption_model_type"] = model_type
            items[idx]["caption_prompt"] = prompt_name
            items[idx]["caption_source"] = "representative_window"
            items[idx]["caption_source_frame_id"] = items[representatives[key]].get("frame_id")
            items[idx]["caption_window_seconds"] = window_seconds
            items[idx]["caption_window_id"] = f"{key[0]}:{key[1]}"
            items[idx]["caption_word_count"] = len(caption.split())
            updated += 1
    return updated


def _load_images(items: list[dict], indices: Iterable[int]) -> tuple[list[Image.Image], list[int]]:
    images = []
    loaded_indices = []

    for idx in indices:
        img_path = resolve_path(items[idx].get("path", ""))
        try:
            with Image.open(img_path) as fh:
                images.append(fh.convert("RGB"))
            loaded_indices.append(idx)
        except Exception as exc:
            logger.warning("Could not read image %s: %s", img_path, exc)

    return images, loaded_indices


def _resolve_model_type(model_name: str, model_type: str) -> str:
    if model_type != "auto":
        return model_type
    if "blip2" in model_name.lower():
        return "blip2"
    return "blip"


def _load_caption_model(model_name: str, model_type: str):
    model_type = _resolve_model_type(model_name, model_type)
    dtype = torch.float16 if str(DEVICE).startswith("cuda") else torch.float32

    if model_type == "blip2":
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    elif model_type == "blip":
        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.to(DEVICE)
    model.eval()
    return model, processor, model_type


def _clean_caption(text: str, prompt: str) -> str:
    text = " ".join(str(text or "").replace("\n", " ").split())
    if not text:
        return ""

    normalized_prompt = " ".join(prompt.split()).strip()
    if normalized_prompt and text.lower().startswith(normalized_prompt.lower()):
        text = text[len(normalized_prompt):].strip(" :-")

    return text.strip()


def generate_keyframe_captions(
    batch_size: int = 2,
    save_every: int = 10,
    limit: int = 0,
    force: bool = False,
    min_words: int = 8,
    model_name: str = "Salesforce/blip-image-captioning-base",
    model_type: str = "blip",
    prompt_name: str = "aic",
    max_new_tokens: int = 48,
    num_beams: int = 2,
    window_seconds: float = 5.0,
) -> None:
    items = _read_metadata()
    if not items:
        return

    prompt = PROMPTS[prompt_name]
    pending, windows, representatives = _caption_windows(
        items,
        window_seconds=window_seconds,
        force=force,
        min_words=min_words,
        limit=limit,
    )
    if not pending:
        propagated = _propagate_window_captions(
            items,
            windows,
            representatives,
            model_name=model_name,
            model_type=model_type,
            prompt_name=prompt_name,
            window_seconds=window_seconds,
            force=force,
        )
        if propagated:
            _flush_to_disk(items)
        logger.info("No new representative captions needed; propagated %d captions.", propagated)
        return

    logger.info(
        "Captioning %d representative windows for %d keyframes with %s on %s (window=%.1fs).",
        len(pending),
        len(items),
        model_name,
        DEVICE,
        window_seconds,
    )
    model, processor, resolved_model_type = _load_caption_model(model_name, model_type)

    updated_count = 0
    batches_since_save = 0

    for start in range(0, len(pending), batch_size):
        batch_indices = pending[start : start + batch_size]
        images, loaded_indices = _load_images(items, batch_indices)
        if not images:
            continue

        inputs = processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=1.1,
                repetition_penalty=1.2,
            )
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for idx, raw_caption in zip(loaded_indices, generated_texts):
            caption = _clean_caption(raw_caption, prompt)
            if len(caption.split()) < min_words:
                logger.debug("Skipping weak caption for %s: %r", items[idx].get("path"), caption)
                continue

            items[idx]["caption"] = caption
            items[idx]["caption_model"] = model_name
            items[idx]["caption_model_type"] = resolved_model_type
            items[idx]["caption_prompt"] = prompt_name
            items[idx]["caption_word_count"] = len(caption.split())
            updated_count += 1

        batches_since_save += 1
        logger.info("Captioned %d/%d pending keyframes.", min(start + batch_size, len(pending)), len(pending))

        if batches_since_save >= save_every:
            _flush_to_disk(items)
            batches_since_save = 0
            logger.info("Saved caption checkpoint to %s.", METADATA_PATH)

    propagated = _propagate_window_captions(
        items,
        windows,
        representatives,
        model_name=model_name,
        model_type=resolved_model_type,
        prompt_name=prompt_name,
        window_seconds=window_seconds,
        force=force,
    )
    _flush_to_disk(items)
    logger.info(
        "Done. Generated %d representative captions and propagated %d window captions to %s.",
        updated_count,
        propagated,
        METADATA_PATH,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual captions for keyframes")
    parser.add_argument("--batch-size", type=int, default=2, help="Representative images per captioning batch")
    parser.add_argument("--save-every", type=int, default=10, help="Flush metadata after every N batches")
    parser.add_argument("--limit", type=int, default=0, help="Limit pending keyframes for a smoke test (0=all)")
    parser.add_argument("--force", action="store_true", help="Regenerate captions even when one already exists")
    parser.add_argument("--min-words", type=int, default=8, help="Minimum words for a caption to be considered usable")
    parser.add_argument("--model-name", default="Salesforce/blip-image-captioning-base", help="Hugging Face caption model")
    parser.add_argument("--model-type", choices=["auto", "blip", "blip2"], default="blip", help="Caption model family")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="aic", help="Prompt preset")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Maximum generated caption tokens")
    parser.add_argument("--num-beams", type=int, default=2, help="Beam search width")
    parser.add_argument("--window-seconds", type=float, default=5.0, help="Seconds per representative caption window")
    args = parser.parse_args()

    generate_keyframe_captions(
        batch_size=args.batch_size,
        save_every=args.save_every,
        limit=args.limit,
        force=args.force,
        min_words=args.min_words,
        model_name=args.model_name,
        model_type=args.model_type,
        prompt_name=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        window_seconds=args.window_seconds,
    )
