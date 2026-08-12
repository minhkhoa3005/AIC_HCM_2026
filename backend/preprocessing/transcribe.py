"""Sinh transcript (kèm timestamp) cho video bằng faster-whisper."""
from pathlib import Path
from typing import List, Dict
from functools import lru_cache

from backend.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
)


@lru_cache(maxsize=1)
def _load_model():
    from faster_whisper import WhisperModel
    return WhisperModel(
        WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
    )


def transcribe_video(video_path: Path) -> List[Dict]:
    """Trả về danh sách segment: {start, end, text} theo giây."""
    model = _load_model()
    segments, _info = model.transcribe(
        str(video_path),
        language=WHISPER_LANGUAGE,
        task="translate",
        vad_filter=True,
    )
    return [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
    ]


