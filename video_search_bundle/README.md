# Video Search Bundle

This folder is the read-only handoff from `video-search-agent` to the LLM
retrieval pipeline. It contains no source videos, keyframes, preprocessing
code, or training data.

## Bundle

`clip-b32-btc-v1/` contains:

- `artifact_manifest.json`
- `video.index`
- `index_metadata.json`
- `lora_weights.pt`

The bundle is a keyframe-level FAISS inner-product index. The LLM project must
use a compatible CLIP ViT-B/32 text encoder and the text-only LoRA checkpoint
declared by the manifest.

Set the LLM project's `.env` to:

```env
AIC_ARTIFACT_DIR=video_search_bundle/clip-b32-btc-v1
AIC_USE_LORA=true
```

When the video-search pipeline rebuilds its bundle, copy these four files again
to this folder. Do not edit `video.index` or `index_metadata.json` manually.
