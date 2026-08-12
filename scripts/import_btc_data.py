"""Script entrypoint cho luồng import BTC keyframes & metadata.

Delegate trực tiếp sang module chuẩn backend.preprocessing.import_btc_data
để đảm bảo không bị phân rác hay lệch logic giữa 2 file.
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.preprocessing.import_btc_data import build_btc_metadata  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrypoint: Import BTC AIC keyframes & metadata"
    )
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