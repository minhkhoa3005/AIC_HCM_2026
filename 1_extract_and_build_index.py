"""Pipeline Batch Run - Ensemble 3 Models (Zero-shot) + FAISS Local

Kiến trúc:
1. Trích xuất đặc trưng (Feature Extraction): CLIP ViT-L/14 + BLIP-2 Vision + BEiT3-base.
2. Không sử dụng LoRA (Zero-shot hoàn toàn).
3. Hợp nhất 3 vector thành vector 2304 chiều.
4. Xây dựng chỉ mục FAISS Local và lưu xuống đĩa.
5. Truy vấn bằng vector CLIP x3 (Phương án tạm thời để khớp chiều dữ liệu).
"""

import json
import logging
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Import models
import open_clip
from transformers import BlipVisionModel, BlipProcessor, BeitModel, BeitImageProcessor
import faiss
from deep_translator import GoogleTranslator

from backend.config import (
    METADATA_PATH,
    BTC_CLIP_FEATURES_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    DEVICE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thông số mô hình
DIM_CLIP = 768
DIM_BLIP = 768 # Sử dụng BLIP Base (768 chiều) thay vì BLIP-2
DIM_BEIT = 768
# Tổng chiều sẽ là 768 + 768 + 768 = 2304.


def step1_extract_features(limit: int = 1000):
    """Bước 1: Trích xuất tuần tự 3 model để tránh OOM."""
    logger.info("=== BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG ENSEMBLE (3 MODELS) ===")
    
    # 1. Đọc danh sách ảnh
    items = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            if line.strip():
                items.append(json.loads(line))
                
    if not items:
        logger.warning("Không có ảnh nào để xử lý.")
        return

    # Khởi tạo thư mục lưu
    out_dir = BTC_CLIP_FEATURES_DIR / "ensemble_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary lưu trữ các vector tạm thời
    # Structure: { video_id_frame_id: [vec_clip, vec_blip, vec_beit] }
    all_features = {}
    for item in items:
        key = f"{item['video_id']}_{item.get('frame_id', 0)}"
        all_features[key] = []
        
    device = DEVICE

    # Khởi tạo đồng thời 3 mô hình
    logger.info("-> Nạp đồng thời 3 mô hình (CLIP, BLIP Base, BEiT v1) vào VRAM...")
    
    # 1. CLIP
    model_clip, _, preprocess_clip = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
    model_clip.eval()
    
    # 2. BLIP
    processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_blip = BlipVisionModel.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to(device)
    model_blip.eval()
    
    # 3. BEiT
    processor_beit = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
    model_beit = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k").to(device)
    model_beit.eval()

    logger.info("-> Bắt đầu trích xuất song song qua 1 vòng lặp...")
    with torch.no_grad():
        for i, item in enumerate(items):
            if i > 0 and i % 500 == 0:
                logger.info(f"Đang xử lý: Đã chạy {i}/{len(items)} ảnh...")
                
            key = f"{item['video_id']}_{item.get('frame_id', 0)}"
            img_path = Path(item["path"])
            
            # Auto-save feature, nếu có rồi thì skip
            feat_path = out_dir / f"{key}.npy"
            if feat_path.exists(): continue
                
            if not img_path.exists(): continue
            
            try:
                img = Image.open(img_path).convert("RGB")
                
                # 1. CLIP Extract
                img_clip = preprocess_clip(img).unsqueeze(0).to(device)
                feat_clip = model_clip.encode_image(img_clip)
                feat_clip = (feat_clip / feat_clip.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze(0)
                
                # 2. BLIP Extract
                inputs_blip = processor_blip(images=img, return_tensors="pt").to(device, torch.float16)
                outputs_blip = model_blip(**inputs_blip)
                feat_blip = outputs_blip.pooler_output.cpu().numpy().squeeze(0)
                feat_blip = feat_blip / np.linalg.norm(feat_blip)
                
                # 3. BEiT Extract
                inputs_beit = processor_beit(images=img, return_tensors="pt").to(device)
                outputs_beit = model_beit(**inputs_beit)
                feat_beit = outputs_beit.pooler_output.cpu().numpy().squeeze(0)
                feat_beit = feat_beit / np.linalg.norm(feat_beit)
                
                # Nối vector
                super_vec = np.concatenate([feat_clip, feat_blip, feat_beit]).astype(np.float32)
                np.save(feat_path, super_vec) # Auto-save ngay lập tức
                
            except Exception as e:
                logger.warning("Lỗi xử lý ảnh %s: %s", img_path, e)

    # Giải phóng VRAM một lần duy nhất
    del model_clip, preprocess_clip, model_blip, processor_blip, model_beit, processor_beit
    torch.cuda.empty_cache()
    logger.info("Đã trích xuất song song thành công. Đã giải phóng VRAM.")

    # ==========================================
    # HỢP NHẤT (CONCATENATE) & LƯU
    # ==========================================
    # Đếm số lượng vector đã lưu
    saved_count = len(list(out_dir.glob("*.npy")))
    logger.info("Đã tìm thấy hoặc trích xuất thành công %d vector.", saved_count)


def step2_build_faiss_index(limit: int = 1000):
    """Bước 2: Nạp các super vector vào FAISS Local Index."""
    logger.info("=== BƯỚC 2: XÂY DỰNG CHỈ MỤC FAISS LOCAL ===")
    
    out_dir = BTC_CLIP_FEATURES_DIR / "ensemble_features"
    if not out_dir.exists():
        logger.error("Chưa có features, vui lòng chạy bước 1 trước.")
        return None, None

    items = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            if line.strip():
                items.append(json.loads(line))

    vectors = []
    metadata = []
    dim = 0
    
    for idx, item in enumerate(items):
        key = f"{item['video_id']}_{item.get('frame_id', 0)}"
        feat_path = out_dir / f"{key}.npy"
        
        if feat_path.exists():
            vec = np.load(feat_path)
            if dim == 0:
                dim = vec.shape[0]
                logger.info("Phát hiện số chiều của Vector: %d", dim)
                
            vectors.append(vec)
            payload = {
                "id": idx,
                "video_id": item["video_id"],
                "frame_id": item.get("frame_id", 0),
                "pts_time": item.get("pts_time", 0.0),
                "path": item.get("path", "")
            }
            metadata.append(payload)

    if not vectors:
        logger.warning("Không tìm thấy vector nào để đưa vào FAISS.")
        return None, None

    # Khởi tạo và nạp dữ liệu vào FAISS
    vectors_np = np.vstack(vectors).astype(np.float32)
    
    # Chuẩn hóa vector trước khi tìm kiếm Cosine
    faiss.normalize_L2(vectors_np)
    
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)
    
    # Lưu xuống đĩa
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    logger.info("Đã xây dựng xong FAISS Index (%d chiều) với %d keyframes.", dim, len(metadata))
    logger.info("- File Index: %s", FAISS_INDEX_PATH)
    logger.info("- File Meta: %s", FAISS_METADATA_PATH)
    return index, metadata


def step3_query(index, metadata, query_text: str = "một bức ảnh về cái cây"):
    """Bước 3: Truy vấn với văn bản (sử dụng workaround nhân bản)."""
    logger.info("=== BƯỚC 3: TRUY VẤN TÌM KIẾM BẰNG VĂN BẢN ===")
    if index is None: return
    
    # [DỊCH THUẬT] Vi -> En
    logger.info(f"Câu truy vấn gốc (Tiếng Việt): {query_text}")
    try:
        query_en = GoogleTranslator(source='vi', target='en').translate(query_text)
        logger.info(f"Bản dịch sang tiếng Anh (Đưa vào Model): {query_en}")
    except Exception as e:
        logger.warning(f"Lỗi dịch thuật, dùng trực tiếp bản gốc: {e}")
        query_en = query_text
    
    # Tải lại CLIP để sinh vector chữ
    logger.info("-> Chạy Text Encoder (CLIP)...")
    model_clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=DEVICE)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    
    text_tokens = tokenizer([query_en]).to(DEVICE)
    with torch.no_grad():
        text_feat = model_clip.encode_text(text_tokens)
        text_feat = (text_feat / text_feat.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze(0) # [768]
        
    # Vì FAISS Index đang là 2304d, ta phải nhồi thêm số 0 vào sau vector chữ 768d
    pad_dim = index.d - text_feat.shape[0]
    if pad_dim > 0:
        query_super = np.concatenate([text_feat, np.zeros(pad_dim, dtype=np.float32)])
    else:
        query_super = text_feat.astype(np.float32)
        
    # Query FAISS
    query_super = query_super.reshape(1, -1)
    faiss.normalize_L2(query_super)
    
    top_k = 5
    scores, indices = index.search(query_super, top_k)
    
    artifact_dir = Path("search_results").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    md_content = f"# Kết quả Tìm Kiếm bằng Ensemble + FAISS\n\n"
    md_content += f"- **Câu truy vấn (Vi)**: `{query_text}`\n"
    md_content += f"- **Bản dịch (En)**: `{query_en}`\n"
    md_content += f"- **Tổng số chiều vector**: `{index.d}d`\n\n"
    md_content += f"| Top | Score | Tựa đề Video (Dịch Vi) | Frame ID | Thời gian | Hình ảnh |\n"
    md_content += f"| :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for i in range(top_k):
        idx = indices[0][i]
        score = scores[0][i]
        
        if idx < 0 or idx >= len(metadata): continue
        p = metadata[idx]
        
        vid = p.get("video_id", "N/A")
        title_en = p.get("video_title", "Chưa rõ")
        try:
            title_vi = GoogleTranslator(source='en', target='vi').translate(title_en)
        except:
            title_vi = title_en

        fid = p.get("frame_id", 0)
        pts = p.get("pts_time", 0.0)
        img_src_path = Path(p.get("path", ""))

        img_artifact_name = f"search_{vid}_{fid}.jpg"
        img_artifact_path = artifact_dir / img_artifact_name

        if img_src_path.exists():
            try:
                img = Image.open(img_src_path)
                img.save(img_artifact_path)
            except Exception:
                pass

        md_content += f"| **#{i+1:02d}** | `{score:.4f}` | `{title_vi} ({vid})` | `{fid}` | `{pts:.1f}s` | ![{vid}_{fid}]({img_artifact_path.as_uri()}) |\n"

    md_file = artifact_dir / "search_results.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Đã lưu kết quả truy vấn tại: %s", md_file)


def step4_temporal_query(index, metadata, temporal_query_text: str = "một người dẫn chương trình -> giao thông đường phố", max_gap_sec: float = 120.0, top_k_candidates: int = 50, top_k_results: int = 5):
    """Bước 4: Tìm kiếm theo thời gian đa giai đoạn (Multi-stage Temporal Search)."""
    import re
    logger.info("=== BƯỚC 4: TÌM KIẾM THEO THỜI GIAN ĐA GIAI ĐOẠN (TEMPORAL SEARCH) ===")
    if index is None: return

    # 1. Phân rã chuỗi truy vấn
    raw_sub_queries = re.split(r'->|sau đó|then|rồi', temporal_query_text, flags=re.IGNORECASE)
    sub_queries = [sq.strip() for sq in raw_sub_queries if sq.strip()]
    
    if len(sub_queries) < 2:
        logger.warning("Câu truy vấn không có từ nối thời gian. Đang chuyển về Query đơn.")
        step3_query(index, metadata, temporal_query_text)
        return

    logger.info("Phát hiện %d sự kiện nối tiếp trong chuỗi:", len(sub_queries))
    for idx, sq in enumerate(sub_queries):
        logger.info("  [Sự kiện %d]: %s", idx + 1, sq)

    model_clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=DEVICE)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    sub_query_candidates = []
    target_dim = index.d

    for sq in sub_queries:
        try:
            sq_en = GoogleTranslator(source='vi', target='en').translate(sq)
        except Exception:
            sq_en = sq
            
        text_tokens = tokenizer([sq_en]).to(DEVICE)
        with torch.no_grad():
            text_feat = model_clip.encode_text(text_tokens)
            text_feat = (text_feat / text_feat.norm(dim=-1, keepdim=True)).cpu().numpy().squeeze(0)

        if target_dim > text_feat.shape[0]:
            padding = np.zeros(target_dim - text_feat.shape[0], dtype=np.float32)
            query_super = np.concatenate([text_feat, padding]).astype(np.float32)
        else:
            query_super = text_feat.astype(np.float32)

        query_super = query_super.reshape(1, -1)
        faiss.normalize_L2(query_super)

        scores, indices = index.search(query_super, top_k_candidates)
        
        candidates = []
        for i in range(top_k_candidates):
            idx = indices[0][i]
            score = scores[0][i]
            if 0 <= idx < len(metadata):
                item = metadata[idx].copy()
                item["score"] = float(score)
                candidates.append(item)
        sub_query_candidates.append(candidates)

    candidates_A = sub_query_candidates[0]
    candidates_B = sub_query_candidates[1]

    video_to_B = {}
    for item_b in candidates_B:
        vid = item_b["video_id"]
        if vid not in video_to_B:
            video_to_B[vid] = []
        video_to_B[vid].append(item_b)

    temporal_matches = []
    for item_a in candidates_A:
        vid = item_a["video_id"]
        pts_a = item_a.get("pts_time", 0.0)
        score_a = item_a["score"]

        if vid in video_to_B:
            for item_b in video_to_B[vid]:
                pts_b = item_b.get("pts_time", 0.0)
                score_b = item_b["score"]

                if 0 < (pts_b - pts_a) <= max_gap_sec:
                    time_gap = pts_b - pts_a
                    combined_score = score_a + score_b - (time_gap / max_gap_sec) * 0.05
                    temporal_matches.append({
                        "video_id": vid,
                        "event_a": item_a,
                        "event_b": item_b,
                        "time_gap": time_gap,
                        "combined_score": combined_score
                    })

    temporal_matches.sort(key=lambda x: x["combined_score"], reverse=True)

    artifact_dir = Path("search_results").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    md_content = f"# Kết quả Tìm Kiếm Theo Thời Gian (Temporal Search)\n\n"
    md_content += f"- **Chuỗi truy vấn**: `{temporal_query_text}`\n"
    md_content += f"- **Sự kiện A**: `{sub_queries[0]}`\n"
    md_content += f"- **Sự kiện B**: `{sub_queries[1]}`\n"
    md_content += f"- **Giới hạn khoảng cách thời gian**: `{max_gap_sec}s`\n\n"
    md_content += f"| Top | Combined Score | Video ID | Sự kiện A (Frame/Time) | Sự kiện B (Frame/Time) | Khoảng cách |\n"
    md_content += f"| :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for i in range(min(top_k_results, len(temporal_matches))):
        match = temporal_matches[i]
        vid = match["video_id"]
        ea = match["event_a"]
        eb = match["event_b"]
        gap = match["time_gap"]
        c_score = match["combined_score"]

        img_a_path = Path(ea.get("path", ""))
        img_b_path = Path(eb.get("path", ""))
        
        artifact_a = artifact_dir / f"temporal_{vid}_A_{ea.get('frame_id',0)}.jpg"
        artifact_b = artifact_dir / f"temporal_{vid}_B_{eb.get('frame_id',0)}.jpg"

        if img_a_path.exists():
            try: Image.open(img_a_path).save(artifact_a)
            except: pass
        if img_b_path.exists():
            try: Image.open(img_b_path).save(artifact_b)
            except: pass

        md_content += f"| **#{i+1:02d}** | `{c_score:.4f}` | `{vid}` | Frame {ea.get('frame_id')} ({ea.get('pts_time'):.1f}s) | Frame {eb.get('frame_id')} ({eb.get('pts_time'):.1f}s) | `{gap:.1f}s` |\n"

    md_file = artifact_dir / "temporal_search_results.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Đã lưu kết quả Temporal Search tại: %s", md_file)


def main():
    limit = 10  # Đặt bằng 0 để chạy toàn bộ dataset
    
    logger.info("=== HỆ THỐNG SEARCH AIC 2026 (ENSEMBLE ZERO-SHOT) ===")

    # 1. Trích xuất tuần tự
    step1_extract_features(limit=limit)

    # 2. Xây dựng FAISS Index nội bộ
    index, metadata = step2_build_faiss_index(limit=limit)

    # 3. Query Đơn (Single Query)
    step3_query(index, metadata, query_text="một bức ảnh về cái cây")

    # 4. Query Thời Gian (Temporal Query - Multi-stage)
    step4_temporal_query(index, metadata, temporal_query_text="người dẫn chương trình -> giao thông đường phố")


if __name__ == "__main__":
    main()

