from __future__ import annotations

from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

DATA_ROOT = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
AUDIO_WAV_DIR = RAW_DATA_DIR / "audio_wav"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
TRANSCRIPTS_DIR = PROCESSED_DATA_DIR / "transcripts"
SEGMENTS_DIR = PROCESSED_DATA_DIR / "segments"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
ML_DATA_DIR = DATA_ROOT / "ml"

MODELS_ROOT = REPO_ROOT / "models"
NLP_MODELS_DIR = MODELS_ROOT / "nlp"
CV_MODELS_DIR = MODELS_ROOT / "cv"
CV_AUDIO_MODELS_DIR = MODELS_ROOT / "cv_audio"
RESULTS_DIR = REPO_ROOT / "results"

PIPELINE_DIR = SRC_ROOT / "pipeline"
GENERAL_PIPELINE_DIR = PIPELINE_DIR / "general_pipeline"
OPENSMILE_DIR = GENERAL_PIPELINE_DIR / "opensmile"

NOTEBOOK_NLP = REPO_ROOT / "NLP2.ipynb"
NOTEBOOK_TRANSCRIBE = REPO_ROOT / "clean_suicide_chunk.ipynb"


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate
