import numpy as np
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Projection head kèm theo Learnable Temperature (logit_scale)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveProjectionModel(nn.Module):
    """Wrap cả Image Head, Text Head và Learnable Temperature Parameter."""

    def __init__(self, input_dim: int, output_dim: int, init_temperature: float = 0.07):
        super().__init__()
        self.image_head = ProjectionHead(input_dim, output_dim)
        self.text_head = ProjectionHead(input_dim, output_dim)

        # Khởi tạo logit_scale = log(1 / temperature) theo đúng chuẩn OpenAI CLIP
        init_scale = np.log(1.0 / init_temperature)
        self.logit_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        proj_img = self.image_head(img_emb)
        proj_txt = self.text_head(txt_emb)
        return proj_img, proj_txt

    def get_logit_scale(self) -> torch.Tensor:
        """Clamp logit_scale để tránh bùng nổ gradient / loss instability."""
        # Giới hạn scale tương đương temperature trong khoảng [0.01, 100]
        return self.logit_scale.clamp(max=np.log(100.0)).exp()


class TemporalVideoEncoder(nn.Module):
    """Mô hình mã hóa temporal video representation từ chuỗi keyframe features dùng GRU 1-layer + Mean Pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # Mean pooling qua tất cả time steps để tổng hợp thông tin cả đoạn video/scene (khớp [GRU -> pooled])
        pooled = out.mean(dim=1)
        return self.proj(pooled)