import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import ENSEMBLE_EMBED_DIM, EMBED_DIM


class AICProjectionHead(nn.Module):
    """Projection head ép từ Ensemble 2304d xuống 768d."""
    def __init__(self, input_dim: int = ENSEMBLE_EMBED_DIM, output_dim: int = EMBED_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)


class ProjectionHead(nn.Module):
    """Projection head kèm theo Learnable Temperature (logit_scale)."""
    def __init__(self, input_dim: int = ENSEMBLE_EMBED_DIM, output_dim: int = EMBED_DIM, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)


class ContrastiveProjectionModel(nn.Module):
    """Wrap cả Image Head, Text Head và Learnable Temperature Parameter."""
    def __init__(self, input_dim: int = ENSEMBLE_EMBED_DIM, output_dim: int = EMBED_DIM, init_temperature: float = 0.07):
        super().__init__()
        self.image_head = ProjectionHead(input_dim, output_dim)
        self.text_head = ProjectionHead(output_dim, output_dim)

        init_scale = np.log(1.0 / init_temperature)
        self.logit_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        proj_img = self.image_head(img_emb)
        proj_txt = self.text_head(txt_emb)
        return proj_img, proj_txt

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(max=np.log(100.0)).exp()


class TemporalVideoEncoder(nn.Module):
    """Mô hình mã hóa temporal video representation từ chuỗi keyframe features dùng GRU 1-layer + Mean Pooling."""
    def __init__(self, input_dim: int = ENSEMBLE_EMBED_DIM, hidden_dim: int = 512, output_dim: int = EMBED_DIM):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.proj(pooled)