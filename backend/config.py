"""Central configuration for the video retrieval pipeline."""
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

# Tự động nạp file .env từ thư mục gốc dự án
env_file = ROOT_DIR / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

VIDEOS_DIR = ROOT_DIR / "data" / "videos"
INDEX_DIR = ROOT_DIR / "data" / "index"
VIDEO_METADATA_DIR = INDEX_DIR / "metadata_by_video"

# Tạo các thư mục cần thiết
for d in (VIDEOS_DIR, INDEX_DIR, VIDEO_METADATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Path Metadata & FAISS Index
METADATA_PATH = INDEX_DIR / "metadata.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "video.index"
SCENE_FAISS_INDEX_PATH = INDEX_DIR / "scene.index"
FAISS_METADATA_PATH = INDEX_DIR / "index_metadata.json"
SCENE_METADATA_PATH = INDEX_DIR / "scene_metadata.json"
PROJECTION_HEAD_PATH = INDEX_DIR / "projection_head.pt"
TRAIN_PAIRS_PATH = INDEX_DIR / "train_pairs.jsonl"


# BTC Data Directories
KEYFRAMES_DIR = ROOT_DIR / "data" / "keyframes"
BTC_MEDIA_INFO_DIR = ROOT_DIR / "data" / "media-info"
BTC_MAP_KEYFRAMES_DIR = ROOT_DIR / "data" / "map-keyframes"
BTC_OBJECTS_DIR = ROOT_DIR / "data" / "objects"
BTC_CLIP_FEATURES_DIR = ROOT_DIR / "data" / "clip-features"

# Preprocessing / Whisper Transcribe
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "vi")

# Embedding / Ensemble Models
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-L-14")
import torch

def _get_default_device():
    env_dev = os.getenv("DEVICE")
    if env_dev:
        if env_dev.lower() in ("dml", "directml"):
            import torch_directml
            return torch_directml.device()
        return env_dev
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = _get_default_device()
TEMPORAL_CHECKPOINT_PATH = INDEX_DIR / "temporal_encoder.pt"
ENSEMBLE_EMBED_DIM = 2304  # CLIP (768) + BLIP (768) + BEiT (768)
EMBED_DIM = 768
PROJECTED_DIM = 768
KEYFRAME_POSITIONS = (0.25, 0.5, 0.75)
TEXT_EMBED_WEIGHT = 0.65
VISUAL_EMBED_WEIGHT = 0.35

# BTC Keyframe Preprocessing / Smart Cutting
BTC_FEATURE_MODEL_NAME = "ViT-B/32"
BTC_FEATURE_DIM = 512
SMART_CUT_SIMILARITY_THRESHOLD = 0.82
SMART_CUT_MIN_SCENE_KEYFRAMES = 3
SMART_CUT_MAX_SCENE_KEYFRAMES = 30

# Fine-tune / Projection Head
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 8
TRAIN_LR = 1e-4
TEMPERATURE = 0.07
VAL_SPLIT_RATIO = 0.2

# API / Search Parameters
DEFAULT_TOP_K = 20
MAX_TOP_K = 100

# Clarification
AMBIGUITY_MARGIN_THRESHOLD = 0.04
CLARIFICATION_TOP_K = 4
MIN_SCORE_TO_CONSIDER = 0.15
