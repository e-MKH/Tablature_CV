from pathlib import Path

# 프로젝트 루트 (config가 src 안에 있으므로 parent.parent)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 데이터 디렉토리
DATA_ROOT = PROJECT_ROOT / "data"
INPUT_DIR = DATA_ROOT / "input"
PROCESSED_VIDEOS_DIR = DATA_ROOT / "processed" / "videos"
PROCESSED_AUDIOS_DIR = DATA_ROOT / "processed" / "audios"
PROCESSED_SESSIONS_DIR = DATA_ROOT / "processed" / "sessions"
ANALYSIS_VISION_DIR = DATA_ROOT / "analysis" / "vision"
ANALYSIS_AUDIO_DIR = DATA_ROOT / "analysis" / "audio"
ANALYSIS_MATCHED_DIR = DATA_ROOT / "analysis" / "matched"
OUTPUT_DIR = DATA_ROOT / "output"
TEMP_DIR = DATA_ROOT / "temp"
MODELS_DIR = DATA_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "weights.pt"