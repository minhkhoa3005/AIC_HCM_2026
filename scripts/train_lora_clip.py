"""Fine-tune CLIP with LoRA on keyframe AIC data.

Optimizations:
- batch tokenize text in the DataLoader collate step
- avoid per-sample translation unless explicitly requested
- use pinned memory / persistent workers on GPU
- keep temporal order stable with keyframe_ordinal-aware loading
"""
import argparse
import json
import logging
import random
import sys
from functools import partial
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

from backend.config import (  # noqa: E402
    CLIP_MODEL_NAME,
    DEVICE,
    KEYFRAMES_DIR,
    LORA_ALPHA,
    LORA_RANK,
    LORA_WEIGHTS_PATH,
    CURRENT_METADATA_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_LR,
    resolve_path,
)
from backend.embedding.clip_encoder import translate_vi_to_en  # noqa: E402
from backend.training.lora import count_trainable_params, inject_lora, save_lora_weights  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class KeyframeCaptionDataset(Dataset):
    """Return raw text and preprocessed image tensors.

    Tokenization is intentionally deferred to the collate step so it can run on
    a whole batch at once instead of once per sample.
    """

    def __init__(self, items: list[dict], preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img_path = resolve_path(item.get("path", ""))

        try:
            with Image.open(img_path) as img:
                img_tensor = self.preprocess(img.convert("RGB"))
        except Exception:
            img_tensor = torch.zeros(3, 224, 224)

        text = item.get("caption") or item.get("text") or item.get("video_title", "")
        return img_tensor, text


def _collate_batch(batch, tokenize_fn):
    images, texts = zip(*batch)
    return torch.stack(images), tokenize_fn(list(texts), truncate=True)


def _resolve_text(item: dict, translate_text: bool) -> str:
    text = (
        item.get("caption", "").strip()
        or item.get("text", "").strip()
        or item.get("video_title", "").strip()
    )
    if not text:
        return ""
    if translate_text:
        return translate_vi_to_en(text)
    return text


def _load_training_data(
    metadata_path: Path,
    limit: int = 0,
    translate_text: bool = False,
    video_ids: set[str] | None = None,
) -> list[dict]:
    if not metadata_path.exists():
        logger.error("Could not find %s. Run import_btc_data.py first.", metadata_path)
        return []

    items = []
    skipped_path = 0
    skipped_text = 0

    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if video_ids is not None and str(item.get("video_id", "")) not in video_ids:
                continue

            img_path = resolve_path(item.get("path", ""))
            if not img_path.exists():
                fallback = KEYFRAMES_DIR / item.get("video_id", "") / img_path.name
                if fallback.exists():
                    item["path"] = str(fallback)
                else:
                    skipped_path += 1
                    continue

            text = _resolve_text(item, translate_text=translate_text)
            if len(text) < 3:
                skipped_text += 1
                continue

            item["caption"] = text
            items.append(item)

            if limit > 0 and len(items) >= limit:
                break

    if skipped_path:
        logger.warning("Skipped %d samples because the image file was missing.", skipped_path)
    if skipped_text:
        logger.warning("Skipped %d samples because they had no usable text.", skipped_text)

    logger.info("Loaded %d training samples from %s", len(items), metadata_path)
    return items


def compute_clip_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_per_image = logit_scale * (image_features @ text_features.T)
    logits_per_text = logits_per_image.T
    labels = torch.arange(len(image_features), device=image_features.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2.0


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs, accum_steps=1, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    logit_scale = model.logit_scale.exp()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    use_amp = scaler is not None
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"

    for step, (images, texts) in enumerate(pbar):
        images = images.to(device, non_blocking=str(device).startswith("cuda"))
        texts = texts.to(device, non_blocking=str(device).startswith("cuda"))

        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            image_features = model.encode_image(images).float()
            text_features = model.encode_text(texts).float()
            loss = compute_clip_loss(image_features, text_features, logit_scale)
            loss = loss / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    logit_scale = model.logit_scale.exp()
    use_amp = str(device).startswith("cuda")
    autocast_device = "cuda" if use_amp else "cpu"

    for images, texts in dataloader:
        images = images.to(device, non_blocking=use_amp)
        texts = texts.to(device, non_blocking=use_amp)

        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            image_features = model.encode_image(images).float()
            text_features = model.encode_text(texts).float()
            loss = compute_clip_loss(image_features, text_features, logit_scale)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _make_loader(dataset, batch_size, shuffle, num_workers, is_gpu, tokenize_fn):
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=is_gpu,
        drop_last=shuffle,
        collate_fn=partial(_collate_batch, tokenize_fn=tokenize_fn),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(**loader_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP with LoRA")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--rank", type=int, default=LORA_RANK, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=LORA_ALPHA, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=TRAIN_LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE, help="Batch size")
    parser.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument(
        "--translate-text",
        action="store_true",
        help="Translate text to English before training. Slower, but can improve retrieval quality.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. On Windows, 0 is safest; increase on GPU if needed.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing LoRA checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu, cuda, dml")
    parser.add_argument(
        "--video-ids",
        type=str,
        default="",
        help="Comma-separated video IDs for this session (empty=all videos)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if args.device is not None else DEVICE
    is_gpu = str(device).startswith("cuda") or "privateuseone" in str(device)
    if is_gpu:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("Training device: %s", device)

    import clip

    logger.info("Loading CLIP %s on %s...", CLIP_MODEL_NAME, device)
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    model = model.float()

    if args.resume and LORA_WEIGHTS_PATH.exists():
        from backend.training.lora import load_lora_weights

        logger.info("Resuming from %s", LORA_WEIGHTS_PATH)
        load_lora_weights(model, LORA_WEIGHTS_PATH)
    else:
        injected = inject_lora(model, rank=args.rank, alpha=args.alpha)
        logger.info("Injected LoRA into %d layers.", injected)

    trainable, total = count_trainable_params(model)
    logger.info("Trainable parameters: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    selected_video_ids = {value.strip() for value in args.video_ids.split(",") if value.strip()} or None
    items = _load_training_data(
        CURRENT_METADATA_PATH,
        limit=args.limit,
        translate_text=args.translate_text,
        video_ids=selected_video_ids,
    )
    if len(items) < 10:
        logger.error("Not enough training data (%d samples).", len(items))
        return

    video_ids = sorted({item["video_id"] for item in items})
    random.shuffle(video_ids)

    if len(video_ids) >= 3:
        n_val = max(1, int(len(video_ids) * args.val_split))
        val_vids = set(video_ids[:n_val])
        train_items = [it for it in items if it["video_id"] not in val_vids]
        val_items = [it for it in items if it["video_id"] in val_vids]
    else:
        random.shuffle(items)
        n_val = max(1, int(len(items) * args.val_split))
        val_items = items[:n_val]
        train_items = items[n_val:]

    if len(train_items) < args.batch_size:
        logger.error("Not enough training samples (%d < batch_size %d).", len(train_items), args.batch_size)
        return

    logger.info("Train: %d samples | Val: %d samples", len(train_items), len(val_items))

    train_dataset = KeyframeCaptionDataset(train_items, preprocess)
    val_dataset = KeyframeCaptionDataset(val_items, preprocess)
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, args.num_workers),
        is_gpu=is_gpu,
        tokenize_fn=clip.tokenize,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        is_gpu=is_gpu,
        tokenize_fn=clip.tokenize,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if is_gpu else None
    best_val_loss = float("inf")
    patience = 3
    no_improve = 0

    logger.info("=== START LORA TRAINING ===")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.epochs,
            accum_steps=max(1, args.accum_steps),
            scaler=scaler,
        )
        val_loss = validate(model, val_loader, device) if val_items else float("nan")

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            marker = " *best*"
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
                    "translate_text": args.translate_text,
                },
            )
        else:
            no_improve += 1

        logger.info("Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f%s", epoch, args.epochs, train_loss, val_loss, marker)

        if no_improve >= patience:
            logger.info("Early stopping after %d epochs without improvement.", patience)
            break

    logger.info("=== FINISHED LORA TRAINING ===")
    logger.info("Best val loss: %.4f", best_val_loss)
    logger.info("Checkpoint saved: %s", LORA_WEIGHTS_PATH)


if __name__ == "__main__":
    main()
