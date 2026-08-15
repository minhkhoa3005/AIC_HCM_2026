import os
from typing import List, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.embedding.search_engine import VectorSearchEngine
from backend.config import DEVICE, FAISS_INDEX_PATH

app = FastAPI(title="AIC 2026 - Video Search Agent API (Ensemble + Temporal)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = VectorSearchEngine(device=DEVICE)

if os.path.exists("data"):
    app.mount("/videos", StaticFiles(directory="data"), name="videos")


@app.on_event("startup")
def startup_event():
    print(f"🚀 Đang khởi tạo Backend Search Engine trên thiết bị: {DEVICE}...")
    if FAISS_INDEX_PATH.exists():
        search_engine.load_index()
        print("✅ Đã load FAISS Index & Metadata thành công.")
    else:
        print("⚠️ Chưa có file FAISS Index trong data/index/. Vui lòng tạo index trước.")


class ResultItem(BaseModel):
    video_id: str
    video_title: str
    score: float
    pts_time: float
    frame_id: str
    submission: str


class SearchResponse(BaseModel):
    query_vi: str
    query_en: Optional[str] = None
    results: List[ResultItem]


@app.get("/")
def home():
    return {"status": "ok", "message": "API Video Search Agent (Ensemble 2304d + Temporal) đang hoạt động!"}


@app.get("/search", response_model=SearchResponse)
def search(
    query: str = Query(..., description="Mô tả hoặc chuỗi thời gian (dùng -> hoặc 'sau đó')"),
    top_k: int = Query(20, description="Số lượng kết quả"),
    task_type: str = Query("kis", description="Dạng bài: kis, qa, trake"),
    answer: Optional[str] = Query(None, description="Câu trả lời cho dạng Q&A")
):
    if search_engine.index is None:
        return SearchResponse(query_vi=query, results=[])

    # Nếu câu truy vấn có ký tự nối thời gian -> Chuyển sang Temporal Search
    if any(sep in query.lower() for sep in ["->", "sau đó", "then", "rồi"]):
        temp_res = search_engine.search_temporal(query, top_k=top_k)
        results = []
        for match in temp_res.get("matches", []):
            vid = match["video_id"]
            eb = match["event_b"]
            f_id = str(eb.get("frame_id", 0))
            pts = float(eb.get("pts_time", 0.0))
            sub_line = f"{vid}, {f_id}, {answer}" if (task_type == "qa" and answer) else f"{vid}, {f_id}"

            results.append(ResultItem(
                video_id=vid,
                video_title=eb.get("video_title_vi", eb.get("video_title", vid)),
                score=float(match["combined_score"]),
                pts_time=pts,
                frame_id=f_id,
                submission=sub_line
            ))
        return SearchResponse(query_vi=query, results=results)

    # Truy vấn đơn (Single Query)
    res = search_engine.search_single(query, top_k=top_k)
    results = []
    for item in res.get("results", []):
        vid = str(item.get("video_id", "N/A"))
        f_id = str(item.get("frame_id", 0))
        pts = float(item.get("pts_time", 0.0))
        sub_line = f"{vid}, {f_id}, {answer}" if (task_type == "qa" and answer) else f"{vid}, {f_id}"

        results.append(ResultItem(
            video_id=vid,
            video_title=item.get("video_title_vi", item.get("video_title", vid)),
            score=float(item.get("score", 0.0)),
            pts_time=pts,
            frame_id=f_id,
            submission=sub_line
        ))

    return SearchResponse(query_vi=res["query_vi"], query_en=res["query_en"], results=results)