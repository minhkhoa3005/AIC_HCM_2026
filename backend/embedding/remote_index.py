"""Remote Vector Database client and index management using Qdrant."""
import hashlib
import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config import (
    EMBED_DIM,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_PREFER_GRPC,
    QDRANT_URL,
)

logger = logging.getLogger(__name__)

_qdrant_client = None


def get_remote_client():
    """Lazy initialize and return Qdrant client."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    try:
        from qdrant_client import QdrantClient

        if QDRANT_URL:
            logger.info(f"Connecting to Qdrant Remote Cloud at {QDRANT_URL}")
            _qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=QDRANT_PREFER_GRPC,
                timeout=30,
            )
        else:
            logger.info(f"Connecting to Qdrant Remote/Local server at {QDRANT_HOST}:{QDRANT_PORT}")
            _qdrant_client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                api_key=QDRANT_API_KEY,
                timeout=10,
            )
        # Test connection
        _qdrant_client.get_collections()
        return _qdrant_client
    except Exception as e:
        logger.warning(f"Could not connect to external Qdrant server ({e}). Falling back to memory mode.")
        try:
            from qdrant_client import QdrantClient
            _qdrant_client = QdrantClient(":memory:")
            return _qdrant_client
        except Exception as err:
            logger.error(f"Failed to initialize Qdrant memory fallback: {err}")
            raise err


def init_remote_collection(
    collection_name: str = QDRANT_COLLECTION_NAME,
    vector_size: int = EMBED_DIM,
    recreate: bool = False,
) -> bool:
    """Ensure remote collection exists in Qdrant."""
    client = get_remote_client()
    from qdrant_client.http import models

    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection_name in collections:
            if recreate:
                client.delete_collection(collection_name=collection_name)
                logger.info(f"Deleted existing collection '{collection_name}' for re-creation.")
            else:
                logger.info(f"Collection '{collection_name}' already exists.")
                return True

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection '{collection_name}' (dim={vector_size}, distance=COSINE).")
        return True
    except Exception as e:
        logger.error(f"Error initializing Qdrant collection '{collection_name}': {e}")
        return False


def _deterministic_uuid(video_id: str, clip_id: str) -> str:
    """Tạo UUID xác định từ video_id + clip_id.

    Đảm bảo cùng keyframe luôn có cùng ID → upsert idempotent,
    không bị duplicate khi chạy lại, và không bị overwrite dữ liệu
    của batch/video khác.
    """
    key = f"{video_id}::{clip_id}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def upsert_vectors_remote(
    items: List[Dict[str, Any]],
    vectors: np.ndarray,
    collection_name: str = QDRANT_COLLECTION_NAME,
    batch_size: int = 100,
) -> int:
    """Upsert vectors and metadata payload to remote Qdrant database.

    Dùng deterministic UUID từ video_id+clip_id làm point_id để:
    - Idempotent: chạy lại không duplicate
    - Additive: push thêm batch mới không ghi đè batch cũ
    """
    if len(items) != len(vectors):
        raise ValueError(f"Items count ({len(items)}) does not match vectors count ({len(vectors)})")

    client = get_remote_client()
    from qdrant_client.http import models

    vector_dim = vectors.shape[1]
    init_remote_collection(collection_name=collection_name, vector_size=vector_dim)

    total_upserted = 0
    points = []

    for idx, (item, vec) in enumerate(zip(items, vectors)):
        # Deterministic UUID từ video_id + clip_id
        video_id = item.get("video_id", "")
        clip_id = item.get("clip_id", str(idx))
        point_id = _deterministic_uuid(video_id, clip_id)

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload=item,
            )
        )

        if len(points) >= batch_size or idx == len(items) - 1:
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
                total_upserted += len(points)
                logger.info(f"Upserted {total_upserted}/{len(items)} points to remote Qdrant.")
            except Exception as exc:
                logger.error(
                    f"Failed to upsert batch ({total_upserted}-{total_upserted + len(points)}) "
                    f"to '{collection_name}': {exc}"
                )
            points = []

    return total_upserted


def search_remote(
    query_vector: np.ndarray,
    top_k: int = 20,
    collection_name: str = QDRANT_COLLECTION_NAME,
    video_id_filter: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Query remote Qdrant vector database for nearest neighbors."""
    client = get_remote_client()
    from qdrant_client.http import models

    if query_vector.ndim == 2:
        query_vec_list = query_vector[0].tolist()
    else:
        query_vec_list = query_vector.tolist()

    query_filter = None
    if video_id_filter:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id_filter),
                )
            ]
        )

    try:
        # Dùng query_points cho qdrant-client >= 1.7
        response = client.query_points(
            collection_name=collection_name,
            query=query_vec_list,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        hits = response.points

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    **payload,
                    "score": float(hit.score),
                }
            )
        return results
    except Exception as e:
        logger.error(f"Error during Qdrant remote search: {e}")
        return []
