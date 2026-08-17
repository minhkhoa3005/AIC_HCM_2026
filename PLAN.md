# PLAN

## Goal

Keep the repository aligned with the current keyframe-centric pipeline:
custom keyframes -> metadata -> CLIP features -> two-level FAISS index -> API/search.

## Completed

- Keyframe-centric preprocessing is the canonical path.
- `pipeline_batch_run.py` now delegates to maintained entrypoints.
- Local API uses `video.index` and `scene.index` instead of a random `.index` file.
- Training pipeline was optimized for batching and GPU use when available.
- CLIP feature extraction exists for custom keyframes in `scripts/extract_clip_features.py`.
- Caption generation now selects one representative keyframe per 5-second VFR-safe window and propagates its caption within that window.

## Next

| Priority | Task | Outcome |
|---|---|---|
| P0 | Keep docs and notebooks synchronized with current entrypoints | No stale commands in README / notebooks |
| P0 | Keep `requirements.txt` aligned with actual imports | Easier fresh setup |
| P1 | Add a small end-to-end smoke test for import -> feature extract -> index build | Catch pipeline drift early |
| P1 | Document the custom-keyframe workflow clearly | Avoid mixing BTC organizer features with custom features |

## Notes

- Vietnamese text can be used as query input, but CLIP text encoding still runs through English translation by default.
- The standalone extractor in the companion video-extract project produces `keyframes/` and `map-keyframes/`, but not CLIP features.
- Caption window grouping uses `pts_time` first and only falls back to frame identifiers when timestamps are unavailable.
