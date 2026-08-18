"""Promote the successfully trained current batch into cumulative metadata."""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import CURRENT_METADATA_PATH, METADATA_PATH  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge current metadata into cumulative metadata")
    parser.add_argument("--clear-current", action="store_true")
    args = parser.parse_args()

    current_rows = _read_jsonl(CURRENT_METADATA_PATH)
    cumulative_rows = _read_jsonl(METADATA_PATH) if METADATA_PATH.exists() else []
    merged = {(str(row["video_id"]), int(row["frame_id"])): row for row in cumulative_rows}
    for row in current_rows:
        merged[(str(row["video_id"]), int(row["frame_id"]))] = row

    rows = sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("video_id", "")),
            float(row.get("pts_time", 0.0)),
            int(row.get("frame_id", 0)),
        ),
    )
    with METADATA_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Merged {len(current_rows)} current rows into {METADATA_PATH} ({len(rows)} total rows).")

    if args.clear_current:
        CURRENT_METADATA_PATH.unlink()
        print(f"Removed {CURRENT_METADATA_PATH}")


if __name__ == "__main__":
    main()
