import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import open_clip
from deep_translator import GoogleTranslator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
try:
    from backend.config import BTC_CLIP_FEATURES_DIR, DEVICE
except ImportError:
    BTC_CLIP_FEATURES_DIR = Path("data/clip-features")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENSEMBLE_DIR = BTC_CLIP_FEATURES_DIR / "ensemble_features"
CHECKPOINT_PATH = "projection_head_latest.pth"


# 1. Định nghĩa Mạng Neural (Projection Head)
class AICProjectionHead(nn.Module):
    def __init__(self, input_dim=2304, output_dim=768, dropout=0.1):
        super().__init__()
        # Ép từ 2304 chiều (Vision) xuống 768 chiều (Text)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        # x: [Batch, 2304]
        out = self.net(x)
        # Chuẩn hóa (L2 Normalize) giống CLIP
        out = F.normalize(out, p=2, dim=-1)
        return out


# 2. Xây dựng Dataset (Tải vào RAM và tính toán trước)
class AICDataset(Dataset):
    def __init__(self, captions_file, clip_model, tokenizer, device):
        self.data = []
        if not os.path.exists(captions_file):
            raise FileNotFoundError(f"Không tìm thấy file captions: {captions_file}")
            
        with open(captions_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        translator = GoogleTranslator(source='vi', target='en')
        
        logger.info("Đang tính toán trước Vector Ngôn ngữ (Pre-computing Text Features)...")
        for item in raw_data:
            key = item["key"]
            caption_vi = item["caption"]
            npy_path = ENSEMBLE_DIR / f"{key}.npy"
            if npy_path.exists():
                # Load vision
                vec = np.load(npy_path).astype(np.float32)
                
                # Translate
                try:
                    caption_en = translator.translate(caption_vi)
                except Exception:
                    caption_en = caption_vi
                    
                # Encode Text
                with torch.no_grad():
                    tokens = tokenizer([caption_en]).to(device)
                    text_feat = clip_model.encode_text(tokens)
                    text_feat = F.normalize(text_feat, p=2, dim=-1).cpu().squeeze(0)
                    
                self.data.append({
                    "vision": torch.tensor(vec),
                    "text": text_feat
                })
                
        logger.info(f"Đã nạp sẵn vào RAM {len(self.data)} cặp vector (Vision + Text) thành công.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["vision"], item["text"]


# 3. Hàm tính Mất mát Contrastive (InfoNCE Loss)
def contrastive_loss(image_features, text_features, logit_scale):
    """
    Tính loss 2 chiều: Ảnh -> Chữ và Chữ -> Ảnh.
    """
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


# 4. Vòng lặp Huấn luyện chính
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Projection Head cho AIC")
    parser.add_argument("--captions", type=str, default="data/captions_dummy.json", help="Đường dẫn file caption json")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=10, help="Số vòng lặp")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Bật cờ này để nạp lại checkpoint gần nhất và train tiếp")
    args = parser.parse_args()

    device = DEVICE
    
    # Nạp mô hình ngôn ngữ (Text Encoder của CLIP) để tạo target
    logger.info("Đang nạp mô hình ngôn ngữ CLIP (FP16) để làm giám khảo (đóng băng)...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14', 
        pretrained='laion2b_s32b_b82k', 
        precision='fp16', 
        device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    clip_model.eval()

    # Chuẩn bị Dữ liệu
    dataset = AICDataset(args.captions, clip_model, tokenizer, device)
    if len(dataset) == 0:
        logger.error("Dataset trống. Hãy kiểm tra lại thư mục npy và file captions.")
        return
        
    should_drop_last = len(dataset) > args.batch_size
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=should_drop_last)

    # Giải phóng CLIP model khỏi VRAM sau khi đã pre-compute xong text features!
    del clip_model
    torch.cuda.empty_cache()
    logger.info("Đã giải phóng mô hình CLIP khỏi VRAM để nhường chỗ cho huấn luyện.")

    # Khởi tạo Mô hình Projection Head
    model = AICProjectionHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    logit_scale = nn.Parameter(torch.ones([], device=device) * np.log(1 / 0.07))
    optimizer.add_param_group({'params': [logit_scale]})
    
    start_epoch = 0

    # Cơ chế Resume (Huấn luyện tiếp tục)
    if args.resume:
        if os.path.exists(CHECKPOINT_PATH):
            logger.info(f"Phát hiện Checkpoint {CHECKPOINT_PATH}. Đang nạp lại trạng thái (Resume)...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logit_scale.data = checkpoint['logit_scale']
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Nạp thành công! Tiếp tục huấn luyện từ Epoch {start_epoch}")
        else:
            logger.warning(f"Bạn gọi --resume nhưng không tìm thấy {CHECKPOINT_PATH}. Sẽ train từ đầu (Scratch).")
    else:
        logger.info("Chế độ: Huấn luyện TỪ ĐẦU (Train from scratch).")

    logger.info(f"=== BẮT ĐẦU HUẤN LUYỆN ({'RESUME' if args.resume else 'SCRATCH'}) ===")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        
        for batch_idx, (vision_feats, text_features) in enumerate(dataloader):
            vision_feats = vision_feats.to(device)
            text_features = text_features.to(device)
            
            # Đẩy ảnh qua Projection Head
            optimizer.zero_grad()
            projected_vision_features = model(vision_feats)
            
            # Tính Loss
            loss = contrastive_loss(projected_vision_features, text_features, logit_scale.exp())
            
            # Tối ưu hóa (Backprop)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch}/{args.epochs-1}] - Batch [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.4f}")

        # Tính trung bình loss của Epoch
        avg_loss = total_loss / max(1, len(dataloader))
        logger.info(f"--- KẾT THÚC EPOCH {epoch} | Average Loss: {avg_loss:.4f} ---")
        
        # [CƠ CHẾ AUTO-SAVE CHECKPOINT]
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'logit_scale': logit_scale.data,
            'loss': avg_loss
        }, CHECKPOINT_PATH)
        logger.info(f"Đã lưu tự động (Check-point) tại {CHECKPOINT_PATH}")

    logger.info("=== HOÀN TẤT HUẤN LUYỆN ===")

if __name__ == "__main__":
    main()
