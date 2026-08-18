"""Central configuration for the video retrieval pipeline."""
import os
from pathlib import Path


def _is_project_root(path: Path) -> bool:
    return (path / "requirements.txt").is_file() and (path / "backend").is_dir()


def _discover_project_root() -> Path:
    env_root = os.getenv("AIC_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser()
        if candidate.exists():
            return candidate.resolve()

    here = Path(__file__).resolve()
    search_roots = [here.parent, *here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in search_roots:
        if candidate.exists() and _is_project_root(candidate):
            return candidate.resolve()

    return here.parent.parent.resolve()


ROOT_DIR = _discover_project_root()


def _resolve_env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default.resolve()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT_DIR / candidate).resolve()


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

DATA_ROOT = _resolve_env_path("AIC_DATA_ROOT", ROOT_DIR / "data")
ZIP_DIR = _resolve_env_path("AIC_ZIP_DIR", ROOT_DIR / "ZIP")

VIDEOS_DIR = DATA_ROOT / "videos"
INDEX_DIR = DATA_ROOT / "index"
VIDEO_METADATA_DIR = INDEX_DIR / "metadata_by_video"
BUNDLE_DIR = INDEX_DIR / "clip-b32-btc-v1"

# Tạo các thư mục cần thiết
for d in (DATA_ROOT, VIDEOS_DIR, INDEX_DIR, VIDEO_METADATA_DIR, BUNDLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Path Metadata & FAISS Index
METADATA_PATH = INDEX_DIR / "metadata.jsonl"
ARTIFACT_MANIFEST_PATH = BUNDLE_DIR / "artifact_manifest.json"
FAISS_INDEX_PATH = BUNDLE_DIR / "video.index"
SCENE_FAISS_INDEX_PATH = BUNDLE_DIR / "scene.index"
FAISS_METADATA_PATH = BUNDLE_DIR / "index_metadata.json"
SCENE_METADATA_PATH = INDEX_DIR / "scene_metadata.json"
TRAIN_PAIRS_PATH = INDEX_DIR / "train_pairs.jsonl"

# LoRA Fine-tune
LORA_WEIGHTS_PATH = BUNDLE_DIR / "lora_weights.pt"
LORA_RANK = int(os.getenv("LORA_RANK", "4"))
LORA_ALPHA = float(os.getenv("LORA_ALPHA", "1.0"))
USE_LORA = os.getenv("AIC_USE_LORA", os.getenv("USE_LORA", "auto")).lower()  # "auto" | "true" | "false"

# BTC Data Directories
KEYFRAMES_DIR = DATA_ROOT / "keyframes"
BTC_MEDIA_INFO_DIR = DATA_ROOT / "media-info"
BTC_MAP_KEYFRAMES_DIR = DATA_ROOT / "map-keyframes"
BTC_OBJECTS_DIR = DATA_ROOT / "objects"
BTC_CLIP_FEATURES_DIR = DATA_ROOT / "clip-features"

# Preprocessing / Whisper Transcribe
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "vi")
WHISPER_TASK = os.getenv("WHISPER_TASK", "transcribe")

# Embedding / CLIP Model
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B/32")
import torch

def _get_default_device():
    env_dev = os.getenv("DEVICE")
    if env_dev:
        if env_dev.lower() in ("dml", "directml"):
            import torch_directml
            return torch_directml.device()
        return env_dev
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return "cpu"

DEVICE = _get_default_device()
TEMPORAL_CHECKPOINT_PATH = INDEX_DIR / "temporal_encoder.pt"
EMBED_DIM = 512
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
VAL_SPLIT_RATIO = 0.2

# API / Search Parameters
DEFAULT_TOP_K = 20
MAX_TOP_K = 100

# Clarification
AMBIGUITY_MARGIN_THRESHOLD = 0.04
CLARIFICATION_TOP_K = 4
MIN_SCORE_TO_CONSIDER = 0.15

# Remote Vector DB Configuration (Qdrant)
USE_REMOTE_VECTOR_DB = os.getenv("USE_REMOTE_VECTOR_DB", "false").lower() in ("true", "1", "yes")
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "qdrant")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "aic2026_keyframes")
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() in ("true", "1", "yes")

def resolve_path(p: str) -> Path:
    """Chuyển đổi đường dẫn (tương đối/tuyệt đối) về dạng Path chuẩn theo ROOT_DIR."""
    if not p:
        return Path("")
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj
    return (ROOT_DIR / path_obj).resolve()
