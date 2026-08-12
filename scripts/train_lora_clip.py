"""Fine-tune CLIP với LoRA trên dữ liệu keyframe AIC.

Quy trình:
  1. Load CLIP ViT-B/32 + inject LoRA adapters (chỉ ~0.3% params trainable).
  2. Đọc training data: ảnh keyframe + caption text từ metadata.jsonl.
  3. Train contrastive loss (InfoNCE) — chỉ update LoRA weights.
  4. Lưu LoRA checkpoint (~2MB) vào data/index/lora_weights.pt.

Sử dụng:
  python scripts/train_lora_clip.py --epochs 5 --rank 4 --batch-size 32
  python scripts/train_lora_clip.py --epochs 3 --limit 1000   # test nhanh
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import (
    CLIP_MODEL_NAME,
    DEVICE,
    LORA_ALPHA,
    LORA_RANK,
    LORA_WEIGHTS_PATH,
    METADATA_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_LR,
)
from backend.training.lora import (
    count_trainable_params,
    inject_lora,
    save_lora_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KeyframeCaptionDataset(Dataset):
    """Dataset trả về (image_tensor, tokenized_text) cho LoRA training.

    Đọc trực tiếp ảnh .jpg (không dùng pre-extracted features)
    vì LoRA cần forward pass qua CLIP image encoder.
    """

    def __init__(self, items: list[dict], preprocess, tokenize_fn):
        self.items = items
        self.preprocess = preprocess
        self.tokenize_fn = tokenize_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = Path(item["path"])

        # Load và preprocess ảnh
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.preprocess(img)
        except Exception:
            # Fallback: trả về ảnh đen nếu lỗi
            img_tensor = torch.zeros(3, 224, 224)

        # Tokenize text
        text = item.get("caption") or item.get("text") or item.get("video_title", "")
        tokens = self.tokenize_fn([text], truncate=True).squeeze(0)

        return img_tensor, tokens


def _load_training_data(metadata_path: Path, limit: int = 0) -> list[dict]:
    """Đọc metadata.jsonl và lọc các keyframe có caption/text hợp lệ."""
    if not metadata_path.exists():
        logger.error("Không tìm thấy %s. Chạy import_btc_data.py trước.", metadata_path)
        return []

    from backend.config import KEYFRAMES_DIR

    items = []
    skipped_path = 0
    skipped_text = 0
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # Cần có ảnh tồn tại — tự động fix path nếu metadata ghi sai
            img_path = Path(item.get("path", ""))
            if not img_path.exists():
                # Fallback: thử tìm trong data/keyframes/<video_id>/<filename>
                video_id = item.get("video_id", "")
                fallback = KEYFRAMES_DIR / video_id / img_path.name
                if fallback.exists():
                    item["path"] = str(fallback)
                else:
                    skipped_path += 1
                    continue

            # Cần có text (ưu tiên: caption > text > video_title)
            text = (
                item.get("caption", "").strip()
                or item.get("text", "").strip()
                or item.get("video_title", "").strip()
            )
            if len(text) < 3:
                skipped_text += 1
                continue

            items.append(item)

    if skipped_path > 0:
        logger.warning("Bỏ qua %d mẫu do không tìm thấy file ảnh.", skipped_path)
    if skipped_text > 0:
        logger.warning("Bỏ qua %d mẫu do không có text/caption.", skipped_text)

    if limit > 0:
        items = items[:limit]

    logger.info("Loaded %d training samples từ %s", len(items), metadata_path)
    return items


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_clip_loss(image_features, text_features, logit_scale):
    """InfoNCE contrastive loss (giống CLIP gốc)."""
    # Normalize
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Cosine similarity as logits
    logits_per_image = logit_scale * (image_features @ text_features.T)
    logits_per_text = logits_per_image.T

    labels = torch.arange(len(image_features), device=image_features.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2.0


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs, accum_steps=1):
    """Train 1 epoch với gradient accumulation, trả về average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    logit_scale = model.logit_scale.exp()

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    for step, (images, texts) in enumerate(pbar):
        images = images.to(device)
        texts = texts.to(device)

        # Forward pass qua CLIP (LoRA layers tham gia ở đây)
        image_features = model.encode_image(images).float()
        text_features = model.encode_text(texts).float()

        loss = compute_clip_loss(image_features, text_features, logit_scale)
        loss = loss / accum_steps  # Scale loss cho gradient accumulation

        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            # Gradient clipping
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validation: tính average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    logit_scale = model.logit_scale.exp()

    for images, texts in dataloader:
        images = images.to(device)
        texts = texts.to(device)

        image_features = model.encode_image(images).float()
        text_features = model.encode_text(texts).float()

        loss = compute_clip_loss(image_features, text_features, logit_scale)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP với LoRA")
    parser.add_argument("--epochs", type=int, default=5, help="Số epochs")
    parser.add_argument("--rank", type=int, default=LORA_RANK, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=LORA_ALPHA, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=TRAIN_LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (nhỏ cho AMD GPU 4GB VRAM)")
    parser.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số mẫu (0=tất cả)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Tỷ lệ validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Học tiếp từ checkpoint LoRA cũ nếu có")
    parser.add_argument("--device", type=str, default=None,
                        help="Thiết bị chạy (cpu, cuda, dml). Mặc định: auto-detect GPU")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device detection: ưu tiên GPU AMD DirectML > CUDA > CPU
    if args.device is not None:
        device = args.device
    else:
        device = DEVICE  # Auto-detect từ config.py (AMD DirectML nếu có)
    
    dev_str = str(device)
    is_gpu = dev_str.startswith("cuda") or "privateuseone" in dev_str
    logger.info("Thiết bị training: %s (GPU: %s)", device, is_gpu)

    # 1. Load CLIP
    import clip
    logger.info("Loading CLIP %s on %s...", CLIP_MODEL_NAME, device)
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)

    # 2. Inject / Resume LoRA
    if args.resume and LORA_WEIGHTS_PATH.exists():
        from backend.training.lora import load_lora_weights
        logger.info("Resuming from existing checkpoint: %s", LORA_WEIGHTS_PATH)
        load_lora_weights(model, LORA_WEIGHTS_PATH)
    else:
        n_injected = inject_lora(model, rank=args.rank, alpha=args.alpha)
        logger.info("Injected fresh LoRA into %d layers.", n_injected)
        
    trainable, total = count_trainable_params(model)
    logger.info(
        "Trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    # 3. Load training data
    items = _load_training_data(METADATA_PATH, limit=args.limit)
    if len(items) < 10:
        logger.error("Không đủ dữ liệu training (cần ít nhất 10 mẫu, có %d).", len(items))
        return

    # Train/val split
    video_ids = sorted(set(item["video_id"] for item in items))
    random.shuffle(video_ids)

    if len(video_ids) >= 3:
        # Split theo video_id (tránh data leakage)
        n_val = max(1, int(len(video_ids) * args.val_split))
        val_vids = set(video_ids[:n_val])
        train_items = [it for it in items if it["video_id"] not in val_vids]
        val_items = [it for it in items if it["video_id"] in val_vids]
    else:
        # Quá ít video → random split theo sample
        random.shuffle(items)
        n_val = max(1, int(len(items) * args.val_split))
        val_items = items[:n_val]
        train_items = items[n_val:]

    if len(train_items) < args.batch_size:
        logger.error(
            "Không đủ training samples (%d < batch_size %d). Thử tăng --limit hoặc giảm --batch-size.",
            len(train_items), args.batch_size,
        )
        return

    logger.info("Train: %d samples | Val: %d samples", len(train_items), len(val_items))

    # 4. DataLoaders
    train_dataset = KeyframeCaptionDataset(train_items, preprocess, clip.tokenize)
    val_dataset = KeyframeCaptionDataset(val_items, preprocess, clip.tokenize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if is_gpu else 0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if is_gpu else 0,
        pin_memory=False,
    )

    # 5. Optimizer (chỉ update LoRA params)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    # 6. Training loop
    best_val_loss = float("inf")
    patience = 3
    no_improve = 0

    logger.info("=== BẮT ĐẦU LORA TRAINING ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs, accum_steps=args.accum_steps)

        val_loss = validate(model, val_loader, device) if val_items else float("nan")

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            marker = " *best*"

            # Lưu checkpoint tốt nhất
            save_lora_weights(
                model,
                LORA_WEIGHTS_PATH,
                metadata={
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "clip_model": CLIP_MODEL_NAME,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "n_train": len(train_items),
                    "n_val": len(val_items),
                },
            )
        else:
            no_improve += 1

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f%s",
            epoch, args.epochs, train_loss, val_loss, marker,
        )

        if no_improve >= patience:
            logger.info("Early stopping sau %d epochs không cải thiện.", patience)
            break

    logger.info("=== HOÀN TẤT LORA TRAINING ===")
    logger.info("Best val loss: %.4f", best_val_loss)
    logger.info("Checkpoint saved: %s", LORA_WEIGHTS_PATH)


if __name__ == "__main__":
    main()
