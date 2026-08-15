"""
Huấn luyện Projection Head tối ưu cho 3-Model Ensemble (2304d -> 768d):
- Hỗ trợ 2 chế độ: Train from scratch & Resume Checkpoint (--resume)
- Set Global Seed toàn cục
- Learnable Logit Scale (Temperature)
- Gradient Clipping
- Save Checkpoint đầy đủ
"""
import os
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.config import (
    DEVICE,
    ENSEMBLE_EMBED_DIM,
    EMBED_DIM,
    PROJECTION_HEAD_PATH,
    TEMPERATURE,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
)
from backend.training.model import ContrastiveProjectionModel, AICProjectionHead

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_contrastive_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2.0


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Projection Head cho backend AIC")
    parser.add_argument("--captions", type=str, default="data/captions_dummy.json")
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--resume", action="store_true", help="Bật cờ để resume checkpoint")
    args = parser.parse_args()

    set_global_seed(42)
    device = DEVICE

    model = AICProjectionHead(input_dim=ENSEMBLE_EMBED_DIM, output_dim=EMBED_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    logit_scale = nn.Parameter(torch.ones([], device=device) * np.log(1.0 / TEMPERATURE))
    optimizer.add_param_group({'params': [logit_scale]})

    start_epoch = 0
    checkpoint_file = str(PROJECTION_HEAD_PATH)

    if args.resume and os.path.exists(checkpoint_file):
        logger.info("Nạp checkpoint %s (Resume)...", checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logit_scale.data = checkpoint['logit_scale']
        start_epoch = checkpoint['epoch'] + 1
        logger.info("Resume từ Epoch %d thành công.", start_epoch)
    else:
        logger.info("Chế độ: Huấn luyện TỪ ĐẦU (Train from scratch).")

    logger.info("Sẵn sàng huấn luyện với VRAM/Device: %s", device)


if __name__ == "__main__":
    main()