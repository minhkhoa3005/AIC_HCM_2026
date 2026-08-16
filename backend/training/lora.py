"""LoRA (Low-Rank Adaptation) module cho CLIP.

Inject LoRA adapters vào các lớp attention của CLIP để fine-tune
mà KHÔNG thay đổi weight gốc → chống Catastrophic Forgetting.

Tham khảo: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

Cách hoạt động:
  - Weight gốc W₀ được FREEZE hoàn toàn.
  - Thêm 2 ma trận nhỏ A (d×r) và B (r×d) với rank r << d.
  - Output = W₀·x + α·(B·A·x), với α = lora_alpha / rank (scaling factor).
  - Chỉ A, B là trainable → ~0.3% tổng params.

Sử dụng:
  >>> import clip
  >>> model, preprocess = clip.load("ViT-B/32", device="cpu")
  >>> inject_lora(model, rank=4, alpha=1.0, target_modules=["out_proj", "c_proj"])
  >>> # Chỉ LoRA params là trainable, phần còn lại frozen
  >>> trainable, total = count_trainable_params(model)
  >>> print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
"""
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Wrap một nn.Linear gốc, thêm LoRA adapter.

    Forward: y = W₀·x + (α/r) · B·A·x
    Chỉ A và B là trainable. W₀ hoàn toàn frozen.

    Lưu ý: cung cấp property .weight và .bias để tương thích với
    torch.nn.MultiheadAttention (truy cập out_proj.weight trực tiếp
    thay vì gọi forward()).
    """

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features
        device = original.weight.device
        dtype = original.weight.dtype

        # A: (in_features, rank) — khởi tạo Kaiming uniform
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        # B: (rank, out_features) — khởi tạo zero để ΔW ban đầu = 0
        # → model bắt đầu training với output giống hệt CLIP gốc
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # Freeze weight gốc
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        """Trả về merged weight (W₀ + ΔW) để tương thích với MHA.

        MultiheadAttention truy cập self.out_proj.weight trực tiếp
        (không qua forward()) và truyền vào F.multi_head_attention_forward.
        Property này đảm bảo gradient vẫn flow qua LoRA params.
        """
        # ΔW = (A @ B)^T * scaling, vì Linear weight có shape (out, in)
        lora_delta = (self.lora_A @ self.lora_B).T * self.scaling
        return self.original.weight + lora_delta

    @property
    def bias(self):
        """Trả về bias gốc (không thay đổi bởi LoRA)."""
        return self.original.bias

    @property
    def in_features(self) -> int:
        return self.original.in_features

    @property
    def out_features(self) -> int:
        return self.original.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output gốc (frozen)
        base_out = self.original(x)

        # LoRA delta: x @ A @ B, scaled
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling

        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.original.in_features}, out={self.original.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: Optional[list[str]] = None,
) -> int:
    """Inject LoRA adapters vào các nn.Linear layers trong CLIP model.

    Args:
        model: CLIP model (từ clip.load()).
        rank: Rank của LoRA decomposition. Nhỏ hơn = ít params hơn.
        alpha: Scaling factor. Thường đặt = rank hoặc 1.0.
        target_modules: Tên các module Linear cần inject. Mặc định chỉ inject
            vào attention projection layers (out_proj, c_proj) — hiệu quả nhất.

    Returns:
        Số lượng layers đã được inject LoRA.
    """
    if target_modules is None:
        # Mặc định: inject vào attention output projection + MLP projection
        # trong cả visual encoder lẫn text encoder
        target_modules = ["out_proj", "c_proj"]

    injected_count = 0

    for name, module in list(model.named_modules()):
        # Bỏ qua toàn bộ Image Encoder để giữ nguyên vector 512d chuẩn
        if "visual" in name:
            continue
            
        # Chỉ inject vào nn.Linear nằm trong target_modules
        if not isinstance(module, nn.Linear):
            continue

        # Lấy tên ngắn (phần cuối) của module
        short_name = name.split(".")[-1]
        if short_name not in target_modules:
            continue

        # Tạo LoRA wrapper
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha)

        # Replace module trong model
        # Cần navigate đến parent module để setattr
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_layer)

        injected_count += 1

    # Freeze tất cả params không phải LoRA
    for param_name, param in model.named_parameters():
        if "lora_" not in param_name:
            param.requires_grad_(False)

    logger.info(
        "Injected LoRA (rank=%d, alpha=%.1f) into %d layers. Target: %s",
        rank, alpha, injected_count, target_modules,
    )
    return injected_count


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Đếm số params trainable vs tổng.

    Returns:
        (trainable_count, total_count)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_lora_weights(model: nn.Module, path: Path, metadata: Optional[dict] = None) -> None:
    """Lưu CHỈ LoRA weights (rất nhỏ, ~2MB thay vì ~600MB).

    File checkpoint chứa:
      - lora_state_dict: chỉ các tham số có "lora_" trong tên
      - rank, alpha: hyperparams để reconstruct
      - metadata: thông tin training bổ sung (optional)
    """
    lora_state = {}
    rank = None
    alpha = None

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
            if rank is None:
                rank = module.rank
                alpha = module.alpha

    if not lora_state:
        logger.warning("Không tìm thấy LoRA layers nào trong model. Không lưu gì cả.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "lora_state_dict": lora_state,
        "rank": rank,
        "alpha": alpha,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    logger.info(
        "Saved LoRA weights (%d tensors, rank=%d) to %s (%.2f KB)",
        len(lora_state), rank, path,
        path.stat().st_size / 1024,
    )


def load_lora_weights(model: nn.Module, path: Path) -> dict:
    """Load LoRA weights đã train vào model.

    Quy trình:
      1. Đọc checkpoint → lấy rank, alpha.
      2. Nếu model chưa có LoRA layers → inject_lora() trước.
      3. Load state dict vào các LoRA layers.

    Returns:
        Metadata dict từ checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    rank = checkpoint["rank"]
    alpha = checkpoint["alpha"]
    lora_state = checkpoint["lora_state_dict"]

    # Kiểm tra xem model đã có LoRA chưa
    has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    if not has_lora:
        inject_lora(model, rank=rank, alpha=alpha)

    # Load weights
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_a = f"{name}.lora_A"
            key_b = f"{name}.lora_B"
            if key_a in lora_state and key_b in lora_state:
                dev = module.lora_A.device
                dt = module.lora_A.dtype
                module.lora_A.data.copy_(lora_state[key_a].to(device=dev, dtype=dt))
                module.lora_B.data.copy_(lora_state[key_b].to(device=dev, dtype=dt))

    logger.info("Loaded LoRA weights (rank=%d, alpha=%.1f) from %s", rank, alpha, path)
    return checkpoint.get("metadata", {})
