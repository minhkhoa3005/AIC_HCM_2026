import json
import os
import glob
from typing import List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import clip
import faiss

app = FastAPI(title="AIC 2026 - Video Search Agent API", version="1.0.0")

# Cấu hình CORS để Frontend kết nối thoải mái
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = None
faiss_index = None
metadata_list = []

# Đăng ký thư mục chứa video (nếu có video thật trong data/videos hoặc data/keyframes)
if os.path.exists("data"):
    app.mount("/videos", StaticFiles(directory="data"), name="videos")

@app.on_event("startup")
def startup_event():
    global clip_model, faiss_index, metadata_list
    print(f"🚀 Đang khởi tạo Backend trên thiết bị: {device}...")
    
    # 1. Load CLIP Model
    clip_model, _ = clip.load("ViT-B/32", device=device)
    print("✅ Đã load CLIP Model (ViT-B/32).")

    # 2. Load Metadata
    meta_path = os.path.join("data", "index", "metadata.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata_list.append(json.loads(line))
        print(f"✅ Đã load {len(metadata_list)} dòng metadata.")
    else:
        print(f"⚠️ Không tìm thấy metadata tại: {meta_path}")

    # 3. Load FAISS Index
    index_files = glob.glob(os.path.join("data", "index", "*.index"))
    if index_files:
        faiss_index = faiss.read_index(index_files[0])
        print(f"✅ Đã load FAISS Index từ: {index_files[0]} ({faiss_index.ntotal} vectors)")
    else:
        print("⚠️ Không tìm thấy file FAISS Index trong data/index/")

class ResultItem(BaseModel):
    video_id: str
    video_title: str
    score: float
    start: float
    end: float
    frame_id: str
    clip_id: str
    text: str
    submission: str

class SearchResponse(BaseModel):
    results: List[ResultItem]
    needs_clarification: bool = False
    clarification: Optional[dict] = None

@app.get("/")
def home():
    return {"status": "ok", "message": "API Video Search Agent đang hoạt động!"}

@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., description="Mô tả sự kiện cần tìm"),
    top_k: int = Query(20, description="Số lượng kết quả"),
    task_type: str = Query("kis", description="Dạng bài: kis, qa, trake"),
    answer: Optional[str] = Query(None, description="Câu trả lời cho dạng Q&A"),
    clarification_answer: Optional[str] = None
):
    if clip_model is None or faiss_index is None:
        return SearchResponse(results=[], needs_clarification=False)

    # 1. Mã hóa câu query bằng CLIP
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        query_vector = text_features.cpu().numpy().astype("float32")

    # 2. Truy vấn FAISS Index
    k_search = min(top_k, faiss_index.ntotal)
    distances, indices = faiss_index.search(query_vector, k_search)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        
        # Lấy thông tin từ metadata (hoặc tạo dữ liệu giả lập nếu thiếu)
        item = metadata_list[idx] if idx < len(metadata_list) else {}
        
        v_id = str(item.get("video_id", f"Video_{idx}"))
        f_id = str(item.get("frame_id", idx * 25))
        pts = float(item.get("pts_time", idx * 1.0))
        text_desc = item.get("text", item.get("tags", f"Khoảnh khắc tại timestamp {pts:.1f}s"))
        
        # Định dạng dòng nộp bài (submission line) theo từng dạng bài thi AIC
        if task_type == "qa" and answer:
            sub_line = f"{v_id}, {f_id}, {answer}"
        else:
            sub_line = f"{v_id}, {f_id}"

        results.append(ResultItem(
            video_id=v_id,
            video_title=item.get("video_title", f"Video {v_id}"),
            score=float(score),
            start=max(0.0, pts - 2.0), # Lùi 2s để xem trước
            end=pts + 3.0,              # Tiến 3s
            frame_id=f_id,
            clip_id=item.get("clip_id", f"clip_{idx}"),
            text=str(text_desc),
            submission=sub_line
        ))

    return SearchResponse(results=results, needs_clarification=False)