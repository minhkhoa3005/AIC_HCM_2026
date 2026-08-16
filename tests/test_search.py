import os
import sys
import json
from pathlib import Path

# Add project root to path so we can import backend
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest
import faiss

TEST_DIM = 256

def test_faiss_build_and_search(tmp_path):
    """
    Kiểm thử thuật toán Cosine Similarity và quy trình Search trên FAISS Local.
    """
    # 1. Chuẩn bị dữ liệu giả lập (3 khung hình)
    metadata = [
        {"id": 0, "video_id": "L22_V001", "frame_id": 100, "text": "con mèo đang ngủ"},
        {"id": 1, "video_id": "L22_V001", "frame_id": 200, "text": "chiếc xe máy đỏ"},
        {"id": 2, "video_id": "L22_V002", "frame_id": 50, "text": "bầu trời xanh"},
    ]
    
    # Tạo 3 vector 256 chiều ngẫu nhiên
    vectors = np.random.rand(3, TEST_DIM).astype(np.float32)
    
    # L2 normalize
    faiss.normalize_L2(vectors)
    
    index = faiss.IndexFlatIP(TEST_DIM)
    index.add(vectors)
    
    index_file = tmp_path / "video.index"
    meta_file = tmp_path / "metadata.json"
    
    faiss.write_index(index, str(index_file))
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)
        
    assert index_file.exists()
    assert index.ntotal == 3
    
    # 3. Thực thi Search với vector truy vấn (Giả lập truy vấn câu "chiếc xe máy đỏ")
    noise = np.random.normal(0, 0.01, TEST_DIM).astype(np.float32)
    query_vec = vectors[1] + noise
    query_super = query_vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_super)
    
    scores, indices = index.search(query_super, 2)
    
    assert len(indices[0]) == 2
    
    top_1_idx = indices[0][0]
    top_1 = metadata[top_1_idx]
    
    assert top_1["video_id"] == "L22_V001"
    assert top_1["text"] == "chiếc xe máy đỏ"
    
    # Điểm score (Cosine) của top 1 phải rất cao (gần bằng 1.0 vì nhiễu rất nhỏ)
    assert scores[0][0] > 0.95, f"Score quá thấp: {scores[0][0]}"
    
    # Điểm top 1 phải lớn hơn top 2
    assert scores[0][0] > scores[0][1]

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


def test_mmr_search(tmp_path):
    """Kiểm thử MMR: Kết quả phải chứa các vector đa dạng, không trùng lặp."""
    from backend.embedding.search_algorithms import mmr_search
    import faiss
    import numpy as np
    
    TEST_DIM = 256
    # Vector 1 và 2 giống hệt nhau, Vector 3 khác biệt
    v1 = np.random.rand(TEST_DIM).astype(np.float32)
    v2 = v1.copy()  # Clone của v1
    v3 = np.random.rand(TEST_DIM).astype(np.float32)
    
    vectors = np.array([v1, v2, v3])
    faiss.normalize_L2(vectors)
    
    index = faiss.IndexFlatIP(TEST_DIM)
    index.add(vectors)
    
    metadata = [{"id": 0, "name": "A"}, {"id": 1, "name": "A_clone"}, {"id": 2, "name": "B"}]
    
    # Query giống v1
    query = v1.copy().reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)
    
    # Nếu MMR hoạt động (lambda=0.5), nó sẽ chọn v1 (hoặc v2) đầu tiên, sau đó chọn v3 làm kết quả thứ 2 
    # thay vì chọn v1 rồi v2 vì v1 và v2 quá giống nhau.
    scores, results = mmr_search(query, index, metadata, top_k=2, lambda_mult=0.3, fetch_k=3)
    
    names = [r["name"] for r in results]
    assert len(names) == 2
    assert "A" in names or "A_clone" in names
    assert "B" in names

def test_rocchio_feedback():
    """Kiểm thử Rocchio: Vector mới dịch chuyển đúng."""
    from backend.embedding.search_algorithms import rocchio_feedback
    import numpy as np
    import faiss
    
    TEST_DIM = 256
    q = np.random.rand(1, TEST_DIM).astype(np.float32)
    faiss.normalize_L2(q)
    
    v_pos = np.random.rand(1, TEST_DIM).astype(np.float32)
    faiss.normalize_L2(v_pos)
    
    v_neg = np.random.rand(1, TEST_DIM).astype(np.float32)
    faiss.normalize_L2(v_neg)
    
    q_new = rocchio_feedback(q, relevant_vectors=[v_pos[0]], non_relevant_vectors=[v_neg[0]], alpha=1.0, beta=1.0, gamma=1.0)
    
    # Góc giữa q_new và v_pos phải nhỏ hơn góc giữa q và v_pos (nghĩa là dot product lớn hơn)
    sim_old_pos = np.dot(q, v_pos.T)[0][0]
    sim_new_pos = np.dot(q_new, v_pos.T)[0][0]
    
    sim_old_neg = np.dot(q, v_neg.T)[0][0]
    sim_new_neg = np.dot(q_new, v_neg.T)[0][0]
    
    assert sim_new_pos > sim_old_pos, "Vector mới phải gần với positive vector hơn"
    assert sim_new_neg < sim_old_neg, "Vector mới phải xa negative vector hơn"
