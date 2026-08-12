"""
Huấn luyện Projection Head tối ưu:
- Khống chế False Negative bằng UniqueVideoBatchSampler
- Set Global Seed toàn cục
- Learnable Logit Scale (Temperature)
- Gradient Clipping
- Lưu đầy đủ Hyperparameters vào Checkpoint
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.config import (
    CLIP_MODEL_NAME,
    DEVICE,
    EMBED_DIM,
    PROJECTED_DIM,
    PROJECTION_HEAD_PATH,
    TEMPERATURE,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
)
from backend.training.dataset import (
    CachedEmbeddingDataset,
    UniqueVideoBatchSampler,
    build_pairs,
    train_val_split,
)
from backend.training.model import ContrastiveProjectionModel


def set_global_seed(seed: int = 42):
    """Thiết lập seed toàn cục đảm bảo khả năng tái lập kết quả (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_contrastive_loss(proj_img, proj_txt, logit_scale):
    """Tính InfoNCE Loss hai chiều sử dụng learnable logit_scale."""
    proj_img = F.normalize(proj_img, dim=-1)
    proj_txt = F.normalize(proj_txt, dim=-1)

    logits = logit_scale * (proj_img @ proj_txt.T)
    labels = torch.arange(len(proj_img), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0


def main():
    # 1. Set seed toàn cục
    set_global_seed(42)

    pairs = build_pairs(use_caption_first=True)
    if len(pairs) < 10:
        print("Không đủ số lượng mẫu hợp lệ để huấn luyện.")
        return

    train_pairs, val_pairs = train_val_split(pairs, seed=42)
    print(f"Train: {len(train_pairs)} pairs | Val: {len(val_pairs)} pairs")

    # 2. Khởi tạo Dataset & Custom Sampler chống False Negative
    train_dataset = CachedEmbeddingDataset(train_pairs)
    n_unique_vids = len({p["video_id"] for p in train_pairs})
    if n_unique_vids >= TRAIN_BATCH_SIZE:
        train_batch_sampler = UniqueVideoBatchSampler(
            pairs=train_pairs,
            batch_size=TRAIN_BATCH_SIZE,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
        )
    else:
        actual_batch_size = min(len(train_pairs), TRAIN_BATCH_SIZE)
        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_batch_size,
            shuffle=True,
        )

    # Validation DataLoader (không cần UniqueVideoBatchSampler)
    val_dataset = CachedEmbeddingDataset(val_pairs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
    ) if val_pairs else None

    # 3. Khởi tạo Model & Optimizer
    model = ContrastiveProjectionModel(
        input_dim=EMBED_DIM,
        output_dim=PROJECTED_DIM,
        init_temperature=TEMPERATURE,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_LR,
        weight_decay=0.01,
    )

    PROJECTION_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    patience = 5
    no_improve = 0
    best_checkpoint = None

    print("\n--- STARTING PROJECTION HEAD TRAINING ---")
    for epoch in range(TRAIN_EPOCHS):
        # === Training Phase ===
        model.train()
        total_loss, n_batches = 0.0, 0

        for img_emb, txt_emb in train_loader:
            img_emb = img_emb.to(DEVICE)
            txt_emb = txt_emb.to(DEVICE)

            proj_img, proj_txt = model(img_emb, txt_emb)
            scale = model.get_logit_scale()

            loss = compute_contrastive_loss(proj_img, proj_txt, scale)

            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping chống loss spike
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # === Validation Phase ===
        val_avg = float("nan")
        if val_loader is not None:
            model.eval()
            val_total, val_batches = 0.0, 0
            with torch.no_grad():
                for img_emb, txt_emb in val_loader:
                    img_emb = img_emb.to(DEVICE)
                    txt_emb = txt_emb.to(DEVICE)
                    proj_img, proj_txt = model(img_emb, txt_emb)
                    scale = model.get_logit_scale()
                    v_loss = compute_contrastive_loss(proj_img, proj_txt, scale)
                    val_total += v_loss.item()
                    val_batches += 1
            val_avg = val_total / max(val_batches, 1)

        current_temp = 1.0 / model.get_logit_scale().item()
        
        marker = ""
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            no_improve = 0
            marker = " *best*"
            # Snapshot state dict ngay tại epoch tốt nhất
            best_checkpoint = {
                "input_dim": EMBED_DIM,
                "projected_dim": PROJECTED_DIM,
                "image_head": {k: v.cpu().clone() for k, v in model.image_head.state_dict().items()},
                "text_head":  {k: v.cpu().clone() for k, v in model.text_head.state_dict().items()},
                "logit_scale": model.logit_scale.data.cpu().clone(),
                "hyperparameters": {
                    "input_dim": EMBED_DIM,
                    "projected_dim": PROJECTED_DIM,
                    "learning_rate": TRAIN_LR,
                    "batch_size": TRAIN_BATCH_SIZE,
                    "epochs": epoch + 1,           # epoch thực tế đã train
                    "initial_temperature": TEMPERATURE,
                    "final_temperature": current_temp,
                    "global_seed": 42,
                    "clip_model_name": CLIP_MODEL_NAME,
                },
            }
        else:
            no_improve += 1

        print(
            f"Epoch {epoch + 1:02d}/{TRAIN_EPOCHS:02d} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_avg:.4f} | "
            f"Temp: {current_temp:.4f}{marker}"
        )

        if no_improve >= patience and val_loader is not None:
            print(f"Early stopping tại epoch {epoch + 1} (không cải thiện sau {patience} epoch).")
            break

    # Sau vòng lặp: lưu checkpoint tốt nhất (hoặc epoch cuối nếu không có val)
    PROJECTION_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_data = best_checkpoint if best_checkpoint is not None else {
        "input_dim": EMBED_DIM,
        "projected_dim": PROJECTED_DIM,
        "image_head": {k: v.cpu() for k, v in model.image_head.state_dict().items()},
        "text_head":  {k: v.cpu() for k, v in model.text_head.state_dict().items()},
        "logit_scale": model.logit_scale.data.cpu(),
        "hyperparameters": {"note": "no_val_split_fallback_to_last_epoch"},
    }
    torch.save(save_data, PROJECTION_HEAD_PATH)
    print(f"\n[SUCCESS] Saved best checkpoint to: {PROJECTION_HEAD_PATH}")


if __name__ == "__main__":
    main()