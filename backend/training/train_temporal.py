"""
Huấn luyện nhánh Temporal Video Encoder (GRU 1 lớp + Mean Pooling) cho retrieval tầng 1.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.config import (
    DEVICE, EMBED_DIM, PROJECTED_DIM,
    TEMPORAL_CHECKPOINT_PATH, 
    TEMPERATURE, TRAIN_BATCH_SIZE, TRAIN_EPOCHS, TRAIN_LR,
)

from backend.training.dataset import (
    TemporalVideoDataset,
    UniqueVideoBatchSampler,
    build_sequence_pairs,
    temporal_collate_fn,
    train_val_split,
)
from backend.training.model import ProjectionHead, TemporalVideoEncoder



def set_global_seed(seed: int = 42):
    """Thiết lập seed toàn cục đảm bảo khả năng tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_contrastive_loss(vid_emb: torch.Tensor, txt_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Tính InfoNCE Loss hai chiều giữa Video Representation (GRU pooled) và Text Embedding."""
    vid_emb = F.normalize(vid_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)

    logits = logit_scale * (vid_emb @ txt_emb.T)
    labels = torch.arange(len(vid_emb), device=logits.device)

    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    return (loss_v2t + loss_t2v) / 2.0


def main():
    set_global_seed(42)

    sequences = build_sequence_pairs()
    if len(sequences) < 5:
        print("Không đủ chuỗi video để huấn luyện nhánh Temporal GRU.")
        return

    train_seqs, val_seqs = train_val_split(sequences, seed=42)
    print(f"Temporal Train: {len(train_seqs)} videos | Val: {len(val_seqs)} videos")

    train_dataset = TemporalVideoDataset(train_seqs)
    # Sử dụng UniqueVideoBatchSampler nếu số lượng video trong train đủ lớn
    if len(train_seqs) >= TRAIN_BATCH_SIZE:
        batch_sampler = UniqueVideoBatchSampler(pairs=train_seqs, batch_size=TRAIN_BATCH_SIZE, drop_last=True)
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, collate_fn=temporal_collate_fn)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(len(train_seqs), TRAIN_BATCH_SIZE),
            shuffle=True,
            collate_fn=temporal_collate_fn,
        )

    # Validation DataLoader
    val_loader = None
    if val_seqs:
        val_dataset = TemporalVideoDataset(val_seqs)
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(len(val_seqs), TRAIN_BATCH_SIZE),
            shuffle=False,
            collate_fn=temporal_collate_fn,
        )

    temporal_encoder = TemporalVideoEncoder(
        input_dim=EMBED_DIM, hidden_dim=512, output_dim=PROJECTED_DIM
    ).to(DEVICE)
    text_head = ProjectionHead(EMBED_DIM, PROJECTED_DIM).to(DEVICE)

    # Learnable Temperature Parameter
    init_scale = np.log(1.0 / TEMPERATURE)
    logit_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32, device=DEVICE))

    optimizer = torch.optim.AdamW(
        list(temporal_encoder.parameters()) + list(text_head.parameters()) + [logit_scale],
        lr=TRAIN_LR,
        weight_decay=0.01,
    )

    TEMPORAL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = 5
    no_improve = 0
    best_checkpoint = None

    print("\n--- STARTING TEMPORAL VIDEO ENCODER TRAINING ---")
    for epoch in range(TRAIN_EPOCHS):
        # === Training Phase ===
        temporal_encoder.train()
        text_head.train()
        total_loss, n_batches = 0.0, 0

        for seq_tensor, txt_emb in train_loader:
            seq_tensor = seq_tensor.to(DEVICE)
            txt_emb = txt_emb.to(DEVICE)

            vid_emb = temporal_encoder(seq_tensor)
            proj_txt = text_head(txt_emb)

            current_scale = logit_scale.clamp(max=np.log(100.0)).exp()
            loss = compute_contrastive_loss(vid_emb, proj_txt, current_scale)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                list(temporal_encoder.parameters()) + list(text_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # === Validation Phase ===
        val_avg = float("nan")
        if val_loader is not None:
            temporal_encoder.eval()
            text_head.eval()
            val_total, val_batches = 0.0, 0
            with torch.no_grad():
                for seq_tensor, txt_emb in val_loader:
                    seq_tensor = seq_tensor.to(DEVICE)
                    txt_emb = txt_emb.to(DEVICE)
                    vid_emb = temporal_encoder(seq_tensor)
                    proj_txt = text_head(txt_emb)
                    current_scale = logit_scale.clamp(max=np.log(100.0)).exp()
                    v_loss = compute_contrastive_loss(vid_emb, proj_txt, current_scale)
                    val_total += v_loss.item()
                    val_batches += 1
            val_avg = val_total / max(val_batches, 1)

        current_temp = 1.0 / logit_scale.clamp(max=np.log(100.0)).exp().item()
    marker = ""
    if val_avg < best_val_loss:
        best_val_loss = val_avg
        no_improve = 0
        marker = " *best*"
        best_checkpoint = {
            "temporal_encoder": {k: v.cpu().clone() for k, v in temporal_encoder.state_dict().items()},
            "text_head":        {k: v.cpu().clone() for k, v in text_head.state_dict().items()},
            "logit_scale": logit_scale.data.cpu().clone(),
            "input_dim": EMBED_DIM,
            "projected_dim": PROJECTED_DIM,
            "hyperparameters": {
                "input_dim": EMBED_DIM,
                "projected_dim": PROJECTED_DIM,
                "learning_rate": TRAIN_LR,
                "batch_size": TRAIN_BATCH_SIZE,
                "epochs": epoch + 1,
            },
        }
    else:
        no_improve += 1

    print(
        f"Epoch {epoch + 1:02d}/{TRAIN_EPOCHS:02d} | "
        f"Temporal Loss: {avg_loss:.4f} | "
        f"Val Loss: {val_avg:.4f} | "
        f"Temp: {current_temp:.4f}{marker}"
    )

    if no_improve >= patience and val_loader is not None:
        print(f"Early stopping tại epoch {epoch + 1}.")
        break

# Lưu sau vòng lặp
TEMPORAL_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
save_data = best_checkpoint if best_checkpoint is not None else {
    "temporal_encoder": {k: v.cpu() for k, v in temporal_encoder.state_dict().items()},
    "text_head":        {k: v.cpu() for k, v in text_head.state_dict().items()},
    "logit_scale": logit_scale.data.cpu(),
    "input_dim": EMBED_DIM,
    "projected_dim": PROJECTED_DIM,
    "hyperparameters": {"note": "no_val_split_fallback_to_last_epoch"},
}
torch.save(save_data, TEMPORAL_CHECKPOINT_PATH)
print(f"\n[SUCCESS] Saved best Temporal checkpoint to: {TEMPORAL_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()