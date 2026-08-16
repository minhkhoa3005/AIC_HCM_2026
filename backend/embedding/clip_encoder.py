"""CLIP encoders shared by indexing, search, training, and evaluation."""
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch

from backend.config import (
    CLIP_MODEL_NAME,
    DEVICE,
    KEYFRAME_POSITIONS,
    LORA_ALPHA,
    LORA_RANK,
    LORA_WEIGHTS_PATH,
    TEXT_EMBED_WEIGHT,
    USE_LORA,
    VISUAL_EMBED_WEIGHT,
)


@lru_cache(maxsize=1)
def _load_clip():
    import clip

    model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    model.eval()

    # Auto-load LoRA nếu có trained weights
    _should_use_lora = (
        (USE_LORA == "auto" and LORA_WEIGHTS_PATH.exists())
        or USE_LORA == "true"
    )
    if _should_use_lora and LORA_WEIGHTS_PATH.exists():
        from backend.training.lora import load_lora_weights

        load_lora_weights(model, LORA_WEIGHTS_PATH)
        model.eval()
        import logging
        logging.getLogger(__name__).info(
            "Loaded LoRA-CLIP from %s (rank=%d, alpha=%.1f)",
            LORA_WEIGHTS_PATH, LORA_RANK, LORA_ALPHA,
        )

    return model, preprocess


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec)
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-8)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norm, 1e-8, None)


def extract_keyframe(clip_path: str) -> Optional[np.ndarray]:
    frames = extract_keyframes(clip_path, positions=(0.5,))
    return frames[0] if frames else None


def extract_keyframes(
    clip_path: str,
    positions: Iterable[float] = KEYFRAME_POSITIONS,
) -> list[np.ndarray]:
    """Extract representative RGB frames from a clip.

    Positions are fractional offsets in [0, 1]. Broken frames are skipped so one
    bad seek does not drop the whole clip.
    """
    cap = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    frames = []
    for pos in positions:
        frame_idx = max(0, min(total - 1, int(total * float(pos))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def encode_image(frame_rgb: np.ndarray) -> np.ndarray:
    from PIL import Image
    model, preprocess = _load_clip()
    img = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(img).float()
    return _normalize(emb.cpu().numpy())[0]


def encode_images(frames_rgb: list[np.ndarray]) -> Optional[np.ndarray]:
    if not frames_rgb:
        return None
    vectors = [encode_image(frame) for frame in frames_rgb]
    return _normalize(np.mean(np.stack(vectors), axis=0))


def encode_clip_visual(clip_path: str) -> Optional[np.ndarray]:
    return encode_images(extract_keyframes(clip_path))


def translate_vi_to_en(text: str) -> str:
    """Tự động dịch Tiếng Việt sang Tiếng Anh để tối ưu hoá cho mô hình CLIP gốc."""
    if not text or not text.strip():
        return text
    
    try:
        from deep_translator import GoogleTranslator
        import re
        
        # Kiểm tra sơ bộ xem có ký tự tiếng Việt không
        if re.search(r'[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', text.lower()):
            translated = GoogleTranslator(source='vi', target='en').translate(text)
            import logging
            logging.getLogger(__name__).info(f"Translated query: '{text}' -> '{translated}'")
            return translated
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Translation failed: {e}. Fallback to original text.")
    
    return text


def encode_text(text: str, translate: bool = True) -> np.ndarray:
    """Encode text qua CLIP."""
    import clip
    model, _ = _load_clip()
    
    if translate:
        text = translate_vi_to_en(text)
        
    text = text.strip() or "empty video segment"
    tok = clip.tokenize([text], truncate=True).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(tok).float()
    return _normalize(emb.cpu().numpy())[0]


def encode_texts(texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
    """Mã hoá hàng loạt chuỗi văn bản theo batch."""
    if not texts:
        return []

    import clip
    model, _ = _load_clip()
    
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = [t.strip() or "empty video segment" for t in texts[i : i + batch_size]]
        tok = clip.tokenize(chunk, truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(tok).float()
            normed = _normalize(emb.cpu().numpy())
            results.extend(normed)
    return results

# Giữ lại alias cho backward compatibility
encode_text_raw = encode_text
encode_texts_raw = encode_texts


def fuse_embeddings(
    text_vec: Optional[np.ndarray],
    visual_vec: Optional[np.ndarray],
    text_weight: float = TEXT_EMBED_WEIGHT,
    visual_weight: float = VISUAL_EMBED_WEIGHT,
) -> np.ndarray:
    if text_vec is None and visual_vec is None:
        raise ValueError("At least one embedding is required")
    if text_vec is None:
        return _normalize(visual_vec)
    if visual_vec is None:
        return _normalize(text_vec)

    fused = (text_weight * text_vec) + (visual_weight * visual_vec)
    return _normalize(fused)
