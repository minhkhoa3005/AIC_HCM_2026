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


# 2. Xây dựng Dataset Đọc Vector từ Đĩa
class AICDataset(Dataset):
    def __init__(self, captions_file):
        self.data = []
        if not os.path.exists(captions_file):
            raise FileNotFoundError(f"Không tìm thấy file captions: {captions_file}")
            
        with open(captions_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        for item in raw_data:
            key = item["key"]
            caption = item["caption"]
            npy_path = ENSEMBLE_DIR / f"{key}.npy"
            if npy_path.exists():
                self.data.append({"path": npy_path, "caption": caption})
                
        logger.info(f"Đã load {len(self.data)} cặp dữ liệu hợp lệ.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        vec = np.load(item["path"]).astype(np.float32)
        return torch.tensor(vec), item["caption"]


# 3. Hàm tính Mất mát Contrastive (InfoNCE Loss)
def contrastive_loss(image_features, text_features, logit_scale):
    """
    Tính loss 2 chiều: Ảnh -> Chữ và Chữ -> Ảnh.
    Hàm này ép các cặp (Ảnh i, Chữ i) lại gần nhau và đẩy các cặp chéo nhau ra xa.
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
    
    # Chuẩn bị Dữ liệu
    dataset = AICDataset(args.captions)
    if len(dataset) == 0:
        logger.error("Dataset trống. Hãy kiểm tra lại thư mục npy và file captions.")
        return
        
    should_drop_last = len(dataset) > args.batch_size
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=should_drop_last)

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

    # Nạp mô hình ngôn ngữ (Text Encoder của CLIP) để tạo target
    # Text Encoder này được GIỮ NGUYÊN (Đóng băng hoàn toàn)
    logger.info("Đang nạp mô hình ngôn ngữ CLIP để làm giám khảo (đóng băng)...")
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    clip_model.eval()

    logger.info(f"=== BẮT ĐẦU HUẤN LUYỆN ({'RESUME' if args.resume else 'SCRATCH'}) ===")
    translator = GoogleTranslator(source='vi', target='en')

    model.train()
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0.0
        
        for batch_idx, (vision_feats, texts_vi) in enumerate(dataloader):
            vision_feats = vision_feats.to(device)
            
            # [Dịch tự động on the fly] Vi -> En
            texts_en = []
            for text_vi in texts_vi:
                try:
                    texts_en.append(translator.translate(text_vi))
                except:
                    texts_en.append(text_vi) # Lỗi thì giữ nguyên
                    
            # Encode Text bằng CLIP (Không tính đạo hàm)
            with torch.no_grad():
                text_tokens = tokenizer(texts_en).to(device)
                text_features = clip_model.encode_text(text_tokens)
                text_features = F.normalize(text_features, p=2, dim=-1)

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
