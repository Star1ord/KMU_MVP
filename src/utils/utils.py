from __future__ import annotations

from .paths import (
    AUDIO_WAV_DIR,
    GENERAL_PIPELINE_DIR,
    ML_DATA_DIR,
    NLP_MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPO_ROOT,
    RESULTS_DIR as RESULTS_PATH,
    ensure_directories,
)

PROJECT_ROOT = REPO_ROOT

DATA_DIR = str(ML_DATA_DIR)
MODELS_DIR = str(NLP_MODELS_DIR)
RESULTS_DIR = str(RESULTS_PATH)
DATA_RAW_DIR = str(RAW_DATA_DIR)
DATA_PROCESSED_DIR = str(PROCESSED_DATA_DIR)
PIPELINE_DIR = str(GENERAL_PIPELINE_DIR)

ensure_directories(
    [
        ML_DATA_DIR,
        NLP_MODELS_DIR,
        RESULTS_PATH,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        AUDIO_WAV_DIR,
    ]
)
