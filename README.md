# AIC 2026 - Video Search Agent

Keyframe-centric video retrieval system for AIC 2026.
The current codebase is organized around custom keyframes, per-frame metadata, CLIP features, and a two-level FAISS index.

## Pipeline

1. Extract BTC data archives.
2. Build `data/keyframes/` and `data/map-keyframes/`.
3. Import metadata into `data/index/metadata.jsonl`.
4. Optionally generate representative captions and transcripts.
5. Extract CLIP features for the keyframes.
6. Build `video.index` and `scene.index`.

## Important Paths

- `data/keyframes/`: extracted keyframes
- `data/map-keyframes/`: frame mapping CSV files
- `data/clip-features/`: `.npy` CLIP features for keyframes
- `data/index/metadata.jsonl`: canonical metadata
- `data/index/video.index`: keyframe-level FAISS index
- `data/index/scene.index`: scene-level FAISS index

## Setup

```bash
pip install -r requirements.txt
```

The code now discovers the project root automatically. If you need to override
paths on another machine, set:

- `AIC_PROJECT_ROOT`: project directory
- `AIC_DATA_ROOT`: data directory
- `AIC_ZIP_DIR`: directory that contains the BTC archives

If you want to use GPU, set `DEVICE=cuda` in `.env`.
For Whisper transcription, `WHISPER_LANGUAGE` defaults to `vi` and `WHISPER_TASK` defaults to `transcribe`.

## Canonical Commands

```bash
python scripts/extract_btc_data.py
python scripts/import_btc_data.py --with-transcript
python scripts/extract_clip_features.py
python -m backend.embedding.build_index
python -m backend.api.main
```

Optional steps:

```bash
python -m backend.preprocessing.generate_captions --window-seconds 5 --batch-size 2 --max-new-tokens 48 --num-beams 2
python scripts/train_lora_clip.py --epochs 3 --batch-size 32 --num-workers 4
```

Caption smoke test before a full run:

```bash
python -m backend.preprocessing.generate_captions --limit 100 --window-seconds 5 --batch-size 2 --prompt aic --max-new-tokens 48 --num-beams 2
```

Captioning is window-based: one representative keyframe is selected per 5-second
window using `pts_time`, captioned with BLIP, and propagated to keyframes in the
same video/time window. This reduces captioning cost while preserving a text label
for CLIP training. Use `--force` to regenerate representatives and their propagated
captions.

## Batch Runner

`pipeline_batch_run.py` now delegates to the canonical entrypoints above.
It no longer keeps its own duplicate training/indexing/search logic.

## Notebook Demos

- `run_pipeline.ipynb`: end-to-end pipeline checklist
- `search_demo.ipynb`: local retrieval demo against `data/index/video.index`

## Query Behavior

The search stack accepts Vietnamese input, translates to English when needed, and then encodes the text with CLIP.
