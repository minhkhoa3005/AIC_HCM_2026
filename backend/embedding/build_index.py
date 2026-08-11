"""Build the default keyframe-centric two-level indexes.

This module intentionally delegates to scripts.build_index_features so the
canonical `python -m backend.embedding.build_index` path uses organizer CLIP
features and creates both scene-level and keyframe-level FAISS indexes.
"""
from scripts.build_index_features import main


if __name__ == "__main__":
    main()
