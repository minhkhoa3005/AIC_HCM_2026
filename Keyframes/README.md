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
`index_metadata.json`. This folder is used by the local/API VLM
reranking/QA stage; CLIP + FAISS retrieval does not require the images.
