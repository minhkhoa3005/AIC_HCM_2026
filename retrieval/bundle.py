"""Optional local FAISS bundle adapter with an injected CLIP text encoder."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from llm.schemas import Candidate
from llm.config import load_project_env
from .models import QueryRetrievalResult

TextEncoder = Callable[[str], np.ndarray]


class BundleError(RuntimeError):
    """Raised when the CLIP bundle is missing or inconsistent."""


class LocalBundleRetriever:
    """Search a validated clip-b32-btc-v1 bundle."""

    def __init__(self, bundle_dir: str | Path, text_encoder: TextEncoder):
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise BundleError("Install faiss-cpu to use LocalBundleRetriever") from exc
        self.faiss = faiss
        self.bundle_dir = Path(bundle_dir)
        self.text_encoder = text_encoder
        manifest_path = self.bundle_dir / "artifact_manifest.json"
        index_path = self.bundle_dir / "video.index"
        metadata_path = self.bundle_dir / "index_metadata.json"
        for path in (manifest_path, index_path, metadata_path):
            if not path.is_file():
                raise BundleError(f"Missing required bundle file: {path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "clip_model": "ViT-B/32",
            "image_embedding_space": "openai_clip_vit_b32",
            "embedding_dimension": 512,
            "metric": "inner_product",
        }
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise BundleError(f"Invalid manifest {key}: {self.manifest.get(key)!r}")
        self.index = self.faiss.read_index(str(index_path))
        if self.index.d != 512 or not isinstance(self.index, self.faiss.IndexFlatIP):
            raise BundleError("video.index must be FAISS IndexFlatIP with dimension 512")
        self.metadata: list[dict[str, Any]] = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.index.ntotal != len(self.metadata) or self.index.ntotal != self.manifest.get("vector_count"):
            raise BundleError("FAISS ntotal, vector_count, and metadata length must match")
        for row in self.metadata:
            if not row.get("video_id") or row.get("frame_id") is None or float(row.get("pts_time", -1)) < 0:
                raise BundleError("Every metadata row needs video_id, original frame_id, and pts_time >= 0")

        adapter = self.manifest.get("text_encoder_adapter", "none")
        lora_path = self.bundle_dir / "lora_weights.pt"
        use_lora = os.getenv("AIC_USE_LORA", "auto").strip().lower()
        if adapter == "text_only_lora":
            if not lora_path.is_file() or use_lora != "true":
                raise BundleError("LoRA manifest requires lora_weights.pt and AIC_USE_LORA=true")
            try:
                import torch
                checkpoint = torch.load(lora_path, map_location="cpu", weights_only=False)
                metadata = checkpoint.get("metadata", {})
                if metadata.get("clip_model") != "ViT-B/32" or metadata.get("adapter_scope") != "text_only":
                    raise BundleError("LoRA checkpoint must declare ViT-B/32 and text_only")
            except ImportError as exc:  # pragma: no cover
                raise BundleError("Install torch to validate the LoRA checkpoint") from exc
        elif lora_path.exists():
            raise BundleError("lora_weights.pt is present but manifest does not declare text_only_lora")

    @classmethod
    def from_env(cls, text_encoder: TextEncoder, bundle_dir: str | Path | None = None) -> "LocalBundleRetriever":
        """Load the bundle location from ``AIC_ARTIFACT_DIR`` or the default path."""
        load_project_env()
        root = bundle_dir or os.getenv("AIC_ARTIFACT_DIR", "data/index/clip-b32-btc-v1")
        return cls(root, text_encoder)

    def search_many(self, queries: list[str], top_k: int = 200) -> list[QueryRetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        results: list[QueryRetrievalResult] = []
        for query_index, query in enumerate(queries):
            vector = np.asarray(self.text_encoder(query), dtype="float32").reshape(1, -1)
            if vector.shape != (1, 512):
                raise BundleError(f"CLIP text encoder returned {vector.shape}; expected (1, 512)")
            self.faiss.normalize_L2(vector)
            scores, indices = self.index.search(vector, min(top_k, self.index.ntotal))
            candidates: list[Candidate] = []
            seen: set[tuple[str, int]] = set()
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                row = self.metadata[int(index)]
                key = (str(row["video_id"]), int(row["frame_id"]))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(Candidate(
                    video_id=key[0],
                    frame_id=key[1],
                    keyframe_path=_resolve_keyframe_path(row, self.bundle_dir),
                    clip_score=float(score),
                    metadata=dict(row),
                ))
            results.append(QueryRetrievalResult(query_index=query_index, query_text=query, candidates=candidates))
        return results


def _resolve_keyframe_path(row: dict[str, Any], bundle_dir: Path) -> str:
    """Resolve metadata paths for optional VLM use without changing metadata."""
    raw = str(row.get("path", row.get("keyframe_path", ""))).strip()
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return str(path)
    root = os.getenv("AIC_KEYFRAMES_ROOT", "").strip()
    if root:
        candidate = Path(root).expanduser() / path
        if candidate.exists():
            return str(candidate.resolve())
    # Keep the original path for diagnostics when images are not bundled.
    return str(path)
