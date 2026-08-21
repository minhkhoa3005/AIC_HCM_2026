"""OpenAI CLIP ViT-B/32 text encoder compatible with the BTC LoRA bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from llm.config import resolve_model_cache_dir


class _LoRALinear(nn.Module):
    """The same linear LoRA wrapper used by video-search-agent."""

    def __init__(self, original: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # CLIP is already placed on the requested device when LoRA is injected.
        # Create adapter parameters there as well; otherwise CUDA inference mixes
        # the CUDA base weights with CPU LoRA matrices.
        device = original.weight.device
        dtype = original.weight.dtype
        self.lora_A = nn.Parameter(
            torch.empty(original.in_features, rank, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, original.out_features, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.original.weight + (self.lora_A @ self.lora_B).T * self.scaling

    @property
    def bias(self) -> torch.Tensor | None:
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def _inject_lora(model: nn.Module, rank: int, alpha: float) -> None:
    for name, module in list(model.named_modules()):
        if "visual" in name or not isinstance(module, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in {"out_proj", "c_proj"}:
            continue
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], _LoRALinear(module, rank, alpha))


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


class ClipTextEncoder:
    """Lazy OpenAI CLIP encoder with optional text-only LoRA."""

    def __init__(self, bundle_dir: str | Path, device: str | None = None) -> None:
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        self.device = self._resolve_device(device)
        self.model = None
        self.clip = None

        manifest_path = self.bundle_dir / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing CLIP manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "clip_model": "ViT-B/32",
            "image_embedding_space": "openai_clip_vit_b32",
            "embedding_dimension": 512,
            "metric": "inner_product",
        }
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise ValueError(f"Invalid bundle manifest {key}: {self.manifest.get(key)!r}")

    @classmethod
    def from_env(cls) -> "ClipTextEncoder":
        bundle = os.getenv("AIC_ARTIFACT_DIR", "video_search_bundle/clip-b32-btc-v1")
        return cls(bundle, os.getenv("AIC_DEVICE", "cpu"))

    @staticmethod
    def _resolve_device(configured: str | None) -> str:
        if configured:
            if configured.lower().startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "AIC_DEVICE=cuda nhưng PyTorch hiện không có CUDA. "
                    "Cài requirements.txt rồi kiểm tra torch.cuda.is_available()."
                )
            return configured
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "Install openai-clip (and a compatible PyTorch build) to encode CLIP text."
            ) from exc

        cache_root = resolve_model_cache_dir()
        download_root = None
        if cache_root is not None:
            clip_cache_dir = cache_root / "clip"
            clip_cache_dir.mkdir(parents=True, exist_ok=True)
            download_root = str(clip_cache_dir)
        model, _ = clip.load(
            "ViT-B/32",
            device=self.device,
            jit=False,
            download_root=download_root,
        )
        adapter = self.manifest.get("text_encoder_adapter", "none")
        use_lora = os.getenv("AIC_USE_LORA", "false").strip().lower() == "true"
        checkpoint_path = self.bundle_dir / "lora_weights.pt"
        if adapter == "text_only_lora":
            if not use_lora or not checkpoint_path.is_file():
                raise ValueError("Bundle requires AIC_USE_LORA=true and lora_weights.pt")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            metadata = checkpoint.get("metadata", {})
            if metadata.get("clip_model") != "ViT-B/32" or metadata.get("adapter_scope") != "text_only":
                raise ValueError("LoRA checkpoint is not a ViT-B/32 text-only adapter")
            _inject_lora(model, int(checkpoint["rank"]), float(checkpoint["alpha"]))
            state = checkpoint["lora_state_dict"]
            for name, module in model.named_modules():
                if isinstance(module, _LoRALinear):
                    module.lora_A.data.copy_(state[f"{name}.lora_A"].to(module.lora_A))
                    module.lora_B.data.copy_(state[f"{name}.lora_B"].to(module.lora_B))
        elif checkpoint_path.exists():
            raise ValueError("lora_weights.pt exists but manifest does not enable text-only LoRA")
        model.eval()
        self.clip = clip
        self.model = model

    def encode_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 512), dtype="float32")
        self._load()
        tokens = self.clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens).float().cpu().numpy()
        embeddings = _normalize(embeddings)
        if embeddings.shape != (len(texts), 512):
            raise ValueError(f"CLIP text encoder returned {embeddings.shape}, expected ({len(texts)}, 512)")
        return embeddings

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_many([text])[0]
