# Keyframes for optional VLM

Place images in this structure:

```text
Keyframes/
├── L21_V001/
│   ├── 001.jpg
│   └── 002.jpg
└── L21_V002/
    └── 001.jpg
```

The filenames and video folders must match the paths referenced by
`index_metadata.json`. This folder is used only by optional Gemini VLM
reranking/QA; CLIP + FAISS retrieval does not require the images.
