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

The build step also creates the submission bundle:

```text
data/index/clip-b32-btc-v1/
  artifact_manifest.json
  video.index
  index_metadata.json
  lora_weights.pt        # only when text_only_lora is enabled
```

`artifact_manifest.json` records the actual `vector_count`. The build fails if
the FAISS count and metadata count differ, if vectors are not CLIP-B/32 sized,
or if a metadata row lacks the original `video_id`, `frame_id`, or non-negative
`pts_time`. Set `AIC_USE_LORA=true` when using the text-only LoRA checkpoint.

## Important Paths

- `data/keyframes/`: extracted keyframes
- `data/map-keyframes/`: frame mapping CSV files
- `data/clip-features/`: `.npy` CLIP features for keyframes
- `data/index/metadata.jsonl`: canonical metadata
- `data/index/clip-b32-btc-v1/`: self-contained organizer bundle
  - `video.index`: keyframe-level FAISS index
  - `index_metadata.json`: metadata in exact FAISS row order
  - `artifact_manifest.json`: bundle contract and vector count

## Setup

```bash
pip install -r requirements.txt
```

The code now discovers the project root automatically. If you need to override
paths on another machine, set:

- `AIC_PROJECT_ROOT`: project directory
- `AIC_DATA_ROOT`: data directory
- `AIC_ZIP_DIR`: directory that contains the BTC archives
- `AIC_ARTIFACT_DIR`: output/input bundle directory; defaults to `data/index/clip-b32-btc-v1`
- `AIC_CLIP_MODEL_NAME`: must remain `ViT-B/32` for the BTC bundle
- `AIC_DEVICE`: `cuda`, `cpu`, or a supported DirectML device

If you want to use GPU, set `AIC_DEVICE=cuda` in `.env`.
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
