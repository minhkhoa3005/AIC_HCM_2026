"""Module trích xuất đặc trưng song song từ 3 mô hình (CLIP ViT-L-14, BLIP Base, BEiT v1)."""
import json
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import open_clip
from transformers import BlipVisionModel, BlipProcessor, BeitModel, BeitImageProcessor

from backend.config import (
    METADATA_PATH,
    BTC_CLIP_FEATURES_DIR,
    DEVICE,
    ENSEMBLE_EMBED_DIM
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsembleFeatureExtractor:
    """Bộ trích xuất siêu vector 2304d từ CLIP + BLIP + BEiT."""
    def __init__(self, device=DEVICE):
        self.device = device
        self.out_dir = BTC_CLIP_FEATURES_DIR / "ensemble_features"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.models_loaded = False

    def load_models(self):
        if self.models_loaded: return
        logger.info("-> Nạp đồng thời 3 mô hình (CLIP ViT-L-14, BLIP Base, BEiT v1) vào VRAM...")
        
        # 1. CLIP ViT-L-14
        self.model_clip, _, self.preprocess_clip = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k', device=self.device
        )
        self.model_clip.eval()
        
        # 2. BLIP Base
        self.processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model_blip = BlipVisionModel.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16
        ).to(self.device)
        self.model_blip.eval()
        
        # 3. BEiT v1
        self.processor_beit = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        self.model_beit = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k").to(self.device)
        self.model_beit.eval()
        
        self.models_loaded = True

    def extract_image(self, img_path: Path) -> np.ndarray:
        """Trích xuất 1 bức ảnh qua 3 model và nối thành vector 2304d."""
        if not self.models_loaded:
            self.load_models()

        img = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            # 1. CLIP
            img_clip = self.preprocess_clip(img).unsqueeze(0).to(self.device)
            feat_clip = self.model_clip.encode_image(img_clip)
            feat_clip = (feat_clip / feat_clip.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze(0)

            # 2. BLIP
            inputs_blip = self.processor_blip(images=img, return_tensors="pt").to(self.device, torch.float16)
            outputs_blip = self.model_blip(**inputs_blip)
            feat_blip = outputs_blip.pooler_output.cpu().numpy().squeeze(0)
            feat_blip = feat_blip / np.linalg.norm(feat_blip)

            # 3. BEiT
            inputs_beit = self.processor_beit(images=img, return_tensors="pt").to(self.device)
            outputs_beit = self.model_beit(**inputs_beit)
            feat_beit = outputs_beit.pooler_output.cpu().numpy().squeeze(0)
            feat_beit = feat_beit / np.linalg.norm(feat_beit)

        super_vec = np.concatenate([feat_clip, feat_blip, feat_beit]).astype(np.float32)
        return super_vec

    def process_dataset(self, limit: int = 0) -> int:
        """Trích xuất hàng loạt theo metadata.jsonl và auto-save xuống đĩa."""
        items = []
        with open(METADATA_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit > 0 and i >= limit: break
                if line.strip(): items.append(json.loads(line))

        if not items:
            logger.warning("Không có dữ liệu metadata để xử lý.")
            return 0

        self.load_models()
        logger.info("-> Bắt đầu trích xuất song song qua 1 vòng lặp...")
        saved_count = 0

        with torch.no_grad():
            for item in items:
                key = f"{item['video_id']}_{item.get('frame_id', 0)}"
                feat_path = self.out_dir / f"{key}.npy"
                if feat_path.exists():
                    saved_count += 1
                    continue

                img_path = Path(item["path"])
                if not img_path.exists(): continue

                try:
                    super_vec = self.extract_image(img_path)
                    np.save(feat_path, super_vec)
                    saved_count += 1
                except Exception as e:
                    logger.warning("Lỗi trích xuất ảnh %s: %s", img_path, e)

        self.unload_models()
        logger.info("Đã xử lý và lưu thành công %d vector (%dd).", saved_count, ENSEMBLE_EMBED_DIM)
        return saved_count

    def unload_models(self):
        if not self.models_loaded: return
        del self.model_clip, self.preprocess_clip, self.model_blip, self.processor_blip, self.model_beit, self.processor_beit
        torch.cuda.empty_cache()
        self.models_loaded = False
        logger.info("Đã dọn dẹp bộ nhớ VRAM.")
