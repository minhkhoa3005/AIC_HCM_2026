import os
import sys
from pathlib import Path

# Add project root to path so we can import backend
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

import backend.embedding.remote_index as remote_index

# Tên collection dùng riêng cho mục đích test
TEST_COLLECTION = "test_qdrant_search"
TEST_DIM = 256

@pytest.fixture(autouse=True)
def setup_qdrant_memory(monkeypatch):
    """
    Mock hàm get_remote_client để ép nó sử dụng Qdrant in-memory.
    Điều này giúp test chạy siêu nhanh và không làm rác database thật.
    """
    memory_client = QdrantClient(":memory:")
    
    def mock_get_client():
        return memory_client
        
    monkeypatch.setattr(remote_index, "get_remote_client", mock_get_client)
    
    # Khởi tạo collection test
    memory_client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config=models.VectorParams(size=TEST_DIM, distance=models.Distance.COSINE),
    )
    yield memory_client


def test_qdrant_upsert_and_search():
    """
    Kiểm thử thuật toán Cosine Similarity và quy trình Search trên Qdrant.
    """
    # 1. Chuẩn bị dữ liệu giả lập (3 khung hình)
    items = [
        {"video_id": "L22_V001", "clip_id": "c1", "frame_id": 100, "text": "con mèo đang ngủ"},
        {"video_id": "L22_V001", "clip_id": "c2", "frame_id": 200, "text": "chiếc xe máy đỏ"},
        {"video_id": "L22_V002", "clip_id": "c1", "frame_id": 50, "text": "bầu trời xanh"},
    ]
    
    # Tạo 3 vector 256 chiều ngẫu nhiên
    vectors = np.random.rand(3, TEST_DIM).astype(np.float32)
    
    # Chuẩn hóa vector thành L2 norm (để Cosine Sim tính chính xác)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # 2. Thực thi Upsert
    inserted_count = remote_index.upsert_vectors_remote(
        items=items,
        vectors=vectors,
        collection_name=TEST_COLLECTION
    )
    assert inserted_count == 3, "Phải insert thành công 3 vectors"

    # 3. Thực thi Search với vector truy vấn (Giả lập truy vấn câu "chiếc xe máy đỏ")
    # Ta lấy chính vector thứ 2 (của xe máy) + 1 chút nhiễu (noise) làm câu truy vấn
    noise = np.random.normal(0, 0.01, TEST_DIM).astype(np.float32)
    query_vec = vectors[1] + noise
    # Normalize lại query
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    # Gọi hàm search
    results = remote_index.search_remote(
        query_vector=query_vec,
        top_k=2,
        collection_name=TEST_COLLECTION
    )
    
    # 4. Kiểm chứng kết quả (Assertions)
    assert len(results) == 2, "Hệ thống phải trả về đúng top_k=2 kết quả"
    
    top_1 = results[0]
    assert top_1["video_id"] == "L22_V001"
    assert top_1["clip_id"] == "c2"
    assert top_1["text"] == "chiếc xe máy đỏ"
    
    # Điểm score (Cosine) của top 1 phải rất cao (gần bằng 1.0 vì nhiễu rất nhỏ)
    assert top_1["score"] > 0.95, f"Score quá thấp: {top_1['score']}"
    
    # Điểm top 1 phải lớn hơn top 2
    assert results[0]["score"] > results[1]["score"]


def test_qdrant_search_with_filter():
    """
    Kiểm thử tính năng Filter theo Video ID.
    """
    # 1. Chuẩn bị dữ liệu giả
    items = [
        {"video_id": "L22_V001", "clip_id": "c1"},
        {"video_id": "L22_V002", "clip_id": "c2"},
    ]
    vectors = np.random.rand(2, TEST_DIM).astype(np.float32)
    
    remote_index.upsert_vectors_remote(
        items=items,
        vectors=vectors,
        collection_name=TEST_COLLECTION
    )
    
    # 2. Search có kèm theo filter L22_V002
    query_vec = np.random.rand(TEST_DIM).astype(np.float32)
    results = remote_index.search_remote(
        query_vector=query_vec,
        top_k=5,
        collection_name=TEST_COLLECTION,
        video_id_filter="L22_V002"
    )
    
    # 3. Đảm bảo toàn bộ kết quả trả về chỉ thuộc L22_V002
    assert len(results) == 1
    assert results[0]["video_id"] == "L22_V002"

def test_vietnamese_translation():
    """
    Kiểm thử chức năng tự động dịch Tiếng Việt sang Tiếng Anh.
    """
    from backend.embedding.clip_encoder import translate_vi_to_en
    
    # Test Tiếng Việt
    vi_query = "một bức ảnh về chiếc xe hơi màu đỏ"
    en_query = translate_vi_to_en(vi_query)
    
    # Kết quả dịch thường sẽ chứa từ 'red car' hoặc 'photo'
    assert "red" in en_query.lower()
    assert "car" in en_query.lower()
    
    # Test Tiếng Anh (không dịch)
    en_original = "A blue sky"
    en_result = translate_vi_to_en(en_original)
    assert en_result == en_original
