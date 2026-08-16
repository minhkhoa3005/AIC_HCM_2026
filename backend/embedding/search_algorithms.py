import numpy as np
import faiss

def mmr_search(query_vector, index, metadata, top_k=10, lambda_mult=0.5, fetch_k=50):
    """
    Maximal Marginal Relevance (MMR) search to diversify kết quả tìm kiếm.
    
    :param query_vector: Numpy array of shape (1, D) (L2 normalized nếu dùng FAISS IP).
    :param index: FAISS index object.
    :param metadata: List các payload metadata.
    :param top_k: Số lượng kết quả cuối cùng trả về.
    :param lambda_mult: Độ ưu tiên cho Relevance (1.0 = Max Relevance, 0.0 = Max Diversity).
    :param fetch_k: Số lượng kết quả fetch thô từ FAISS.
    :return: A tuple of (scores, selected_metadata_items)
    """
    # 1. Fetch top fetch_k results
    scores, indices = index.search(query_vector, fetch_k)
    scores = scores[0]
    indices = indices[0]
    
    # Lọc bỏ các index không hợp lệ (khi index.ntotal < fetch_k)
    valid_mask = indices >= 0
    scores = scores[valid_mask]
    indices = indices[valid_mask]
    
    if len(indices) == 0:
        return [], []
        
    # Reconstruct vectors của các candidate để tính độ đa dạng (similarity với nhau)
    try:
        candidate_vectors = np.array([index.reconstruct(int(idx)) for idx in indices])
    except Exception:
        # Nếu FAISS index không hỗ trợ reconstruct (ví dụ IVFPQ), fallback về search chuẩn
        top_k = min(top_k, len(indices))
        return scores[:top_k], [metadata[i] for i in indices[:top_k]]
        
    # L2 normalize candidate vectors 
    faiss.normalize_L2(candidate_vectors)
    
    # Ma trận similarity giữa các ứng viên
    sim_matrix = np.dot(candidate_vectors, candidate_vectors.T)
    
    # Giải thuật MMR
    selected_indices = []
    unselected_indices = list(range(len(indices)))
    
    # Chọn ứng viên có độ đo cosine cao nhất (Relevance lớn nhất) làm item đầu tiên
    selected_indices.append(unselected_indices.pop(0))
    
    while len(selected_indices) < top_k and len(unselected_indices) > 0:
        best_score = -np.inf
        best_idx_to_select = -1
        
        for idx in unselected_indices:
            # Relevance đối với query (Đã tính sẵn từ FAISS)
            relevance = scores[idx]
            
            # Max similarity đối với các tài liệu ĐÃ CHỌN
            max_sim_to_selected = max([sim_matrix[idx, sel_idx] for sel_idx in selected_indices])
            
            # Điểm MMR
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx_to_select = idx
                
        selected_indices.append(best_idx_to_select)
        unselected_indices.remove(best_idx_to_select)
        
    final_metadata = [metadata[indices[i]] for i in selected_indices]
    final_scores = [scores[i] for i in selected_indices]
    
    return final_scores, final_metadata

def rocchio_feedback(query_vector, relevant_vectors, non_relevant_vectors, alpha=1.0, beta=0.75, gamma=0.15):
    """
    Giải thuật Rocchio Relevance Feedback để điều chỉnh vector tìm kiếm.
    
    :param query_vector: Vector truy vấn ban đầu shape (1, D).
    :param relevant_vectors: List/array các vectors tích cực.
    :param non_relevant_vectors: List/array các vectors tiêu cực.
    :param alpha: Trọng số giữ lại cho câu truy vấn ban đầu.
    :param beta: Trọng số của các vectors tích cực.
    :param gamma: Trọng số phạt các vectors tiêu cực.
    :return: Vector truy vấn mới đã chuẩn hóa L2, shape (1, D).
    """
    new_query = alpha * query_vector.copy()
    
    if len(relevant_vectors) > 0:
        rel_mean = np.mean(relevant_vectors, axis=0)
        new_query += beta * rel_mean
        
    if len(non_relevant_vectors) > 0:
        non_rel_mean = np.mean(non_relevant_vectors, axis=0)
        new_query -= gamma * non_rel_mean
        
    # Resize về dạng (1, D) cho FAISS và chuẩn hóa L2 norm (cần thiết cho Inner Product = Cosine Similarity)
    new_query = new_query.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(new_query)
    
    return new_query


def temporal_search(temporal_query_text: str, encode_fn, index, metadata, max_gap_sec: float = 120.0, top_k_candidates: int = 50, top_k: int = 5):
    """
    Tìm kiếm theo thời gian đa giai đoạn (Multi-stage Temporal Search).
    Chỉ hỗ trợ tối đa 2 vế (A -> B).
    """
    import re
    raw_sub_queries = re.split(r'->|sau đó|then|rồi', temporal_query_text, flags=re.IGNORECASE)
    sub_queries = [sq.strip() for sq in raw_sub_queries if sq.strip()]

    # Fallback về MMR nếu chỉ có 1 câu
    if len(sub_queries) < 2:
        query_vec = encode_fn(temporal_query_text)
        scores, results = mmr_search(query_vec, index, metadata, top_k=top_k, lambda_mult=0.5, fetch_k=50)
        return scores, results, sub_queries

    sub_candidates = []
    for sq in sub_queries[:2]: # Giới hạn 2 vế
        query_vec = encode_fn(sq)
        scores, results = mmr_search(query_vec, index, metadata, top_k=top_k_candidates, lambda_mult=0.5, fetch_k=top_k_candidates * 2)
        
        # Nhúng điểm vào candidate
        for r, s in zip(results, scores):
            r['score'] = s
        sub_candidates.append(results)

    candidates_A, candidates_B = sub_candidates[0], sub_candidates[1]
    
    video_to_B = {}
    for item_b in candidates_B:
        vid = item_b["video_id"]
        if vid not in video_to_B: video_to_B[vid] = []
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
                    
                    match_item = item_b.copy()
                    match_item["temporal_score"] = combined_score
                    match_item["event_a_pts"] = pts_a
                    match_item["event_a_frame"] = item_a.get("frame_id", 0)
                    match_item["time_gap"] = time_gap
                    temporal_matches.append(match_item)

    # Sort và chọn top_k
    temporal_matches.sort(key=lambda x: x["temporal_score"], reverse=True)
    temporal_matches = temporal_matches[:top_k]
    
    final_scores = [x["temporal_score"] for x in temporal_matches]
    return final_scores, temporal_matches, sub_queries[:2]
