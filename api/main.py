from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib, pickle
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load simple test models (kept only for backwards compatibility with test_models.py)
MODEL_PATHS = {
    'nlp': Path('models/nlp/early_fusion_linear_svc.pkl'),
    'cv': Path('models/cv/model.joblib'),
    'cv_audio': Path('models/cv_audio/model.joblib'),
}

models = {}
for name, path in MODEL_PATHS.items():
    if path.exists():
        try:
            # Prefer joblib (handles both joblib files and many pickle-based sklearn dumps).
            # Fall back to pickle if joblib fails for some rare cases.
            try:
                models[name] = joblib.load(path)
            except Exception:
                models[name] = pickle.load(open(path, 'rb'))
            logger.info(f"✓ {name} loaded")
        except Exception as e:
            logger.error(f"✗ {name} error: {type(e).__name__}: {e}")
    else:
        logger.warning(f"⚠ {name} not found at {path}")


@app.get("/health")
def health():
    """Return basic information about loaded test models."""
    return {"loaded": list(models.keys()), "total": len(models)}


# Video upload -> preprocess -> inference endpoint
from fastapi import UploadFile, File, HTTPException, status
from fastapi.concurrency import run_in_threadpool
import tempfile

try:
    # Import here so API starts even if preprocessing deps are missing
    from src.preprocessing.analyze_new_media import analyze_new_media_file
except Exception as e:
    analyze_new_media_file = None
    logger.warning(f"analyze_new_media_file import failed: {e}")

try:
    # Video CV pipeline / model (used by /test/cv)
    from inference.video_ensemble import (
        process_video_for_prediction,
        predict_with_video_model,
    )
except Exception as e:
    process_video_for_prediction = None
    predict_with_video_model = None
    logger.warning(f"video_ensemble import failed: {e}")

try:
    # CV+Audio model (ONNX)
    from src.pipeline.cv_audio_pipeline import predict_cv_audio
except Exception as e:
    predict_cv_audio = None
    logger.warning(f"cv_audio import failed: {e}")


def _validate_video_content_type(file: UploadFile) -> None:
    """Shared validation for video uploads."""
    allowed = {
        "video/mp4",
        "video/mpeg",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
    }
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}",
        )


async def _analyze_uploaded_video(
    file: UploadFile,
    skip_transcription: bool = False,
    label: int | None = None,
) -> dict:
    """Common helper: save uploaded video, run full NLP(+video) pipeline, clean up."""
    if analyze_new_media_file is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server missing preprocessing functionality",
        )

    _validate_video_content_type(file)

    tmp_dir = tempfile.mkdtemp(prefix="api_upload_")
    tmp_path = Path(tmp_dir) / file.filename

    try:
        # Save upload to disk
        with open(tmp_path, "wb") as out:
            content = await file.read()
            out.write(content)

        # Run processing in threadpool to avoid blocking the event loop
        result = await run_in_threadpool(
            analyze_new_media_file,
            str(tmp_path),
            label,
            False,  # add_to_training
            "medium",  # whisper_model
            skip_transcription,
        )
        return result
    finally:
        # Best-effort cleanup
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        try:
            Path(tmp_dir).rmdir()
        except Exception:
            pass


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    skip_transcription: bool = False,
    label: int | None = None,
):
    """
    End-to-end production endpoint.

    Upload a video file, run full preprocessing + NLP (and optional CV) models,
    and return the same structure as in offline prod analyze_new_media_file.
    """
    return await _analyze_uploaded_video(
        file=file,
        skip_transcription=skip_transcription,
        label=label,
    )


@app.post("/test/nlp")
async def test_nlp(
    file: UploadFile = File(...),
    skip_transcription: bool = False,
    label: int | None = None,
):
    """
    NLP-focused endpoint with the same semantics, как продовая модель.

    Принимает видео, прогоняет его через тот же analyze_new_media_file,
    что и /predict (полный пайплайн и продовая late_fusion_v3_clean),
    но возвращает компактный ответ, сфокусированный на NLP-результате.
    """
    raw = await _analyze_uploaded_video(
        file=file,
        skip_transcription=skip_transcription,
        label=label,
    )

    prediction = raw.get("prediction")
    risk_score = raw.get("risk_score")
    risk_level = raw.get("risk_level")

    # 0 = контрольная группа, 1 = экспериментальная (risk)
    if prediction == 1:
        prediction_label = "experimental"  # RISK
    elif prediction == 0:
        prediction_label = "control"  # CONTROL
    else:
        prediction_label = None

    return {
        "success": bool(raw.get("success", False)),
        "prediction": prediction,
        "prediction_label": prediction_label,
        "risk_score": risk_score,
        "risk_level": risk_level,
        # Для дальнейшей визуализации можно вернуть сегменты,
        # но без технических полей вроде features_csv и audio_path.
        "segments": raw.get("segments", []),
    }


@app.post("/test/cv")
async def test_cv(
    file: UploadFile = File(...),
):
    """
    CV-only endpoint: принимает видео, извлекает CV-фичи и прогоняет через видео-модель.

    Использует inference.video_ensemble: process_video_for_prediction + predict_with_video_model.
    Возвращает probability, prediction, risk_level и текстовую метку класса,
    либо error, если CV пайплайн недоступен.
    """
    if process_video_for_prediction is None or predict_with_video_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video CV pipeline not available on this server",
        )

    _validate_video_content_type(file)

    tmp_dir = tempfile.mkdtemp(prefix="api_cv_upload_")
    tmp_path = Path(tmp_dir) / file.filename

    try:
        # Save upload
        with open(tmp_path, "wb") as out:
            content = await file.read()
            out.write(content)

        # 1) Прогоняем видео через CV-пайплайн → CSV с фичами
        csv_path, video_err = process_video_for_prediction(
            video_path=str(tmp_path),
            output_dir=None,
            sample_every=3,
            use_emotions=None,
        )

        if video_err or not csv_path:
            # Возвращаем структурированную ошибку, но не роняем сервер
            return {
                "success": False,
                "stage": "video_processing",
                "error": video_err or "Unknown error during video processing",
            }

        # 2) Делаем предсказание видео-моделью
        cv_result = predict_with_video_model(
            video_csv_path=csv_path,
            model_path=None,
            threshold=0.6,
        )

        # Добавляем человекочитаемую метку класса (0=control, 1=experimental)
        if cv_result.get("success") and cv_result.get("prediction") is not None:
            if int(cv_result["prediction"]) == 1:
                cv_result["prediction_label"] = "experimental"
            else:
                cv_result["prediction_label"] = "control"

        # Добавляем для отладки путь к CSV (можно убрать в проде)
        cv_result.setdefault("csv_path", csv_path)
        return cv_result

    finally:
        # Чистим временные файлы
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        try:
            Path(tmp_dir).rmdir()
        except Exception:
            pass


@app.post("/predict/cv-audio")
async def predict_cv_audio_endpoint(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    sample_rate: float = 1.0,
):
    """
    CV+Audio endpoint: принимает видео, извлекает лица (RetinaFace), 
    VGG16 визуальные фичи + аудио фичи, и прогоняет через CV+Audio модель.
    
    Возвращает probability, prediction, prediction_label и risk_level.
    """
    if predict_cv_audio is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CV+Audio pipeline not available on this server",
        )

    _validate_video_content_type(file)

    tmp_dir = tempfile.mkdtemp(prefix="api_cv_audio_upload_")
    tmp_path = Path(tmp_dir) / file.filename

    try:
        # Save upload
        with open(tmp_path, "wb") as out:
            content = await file.read()
            out.write(content)

        # Run CV+Audio prediction
        result = await run_in_threadpool(
            predict_cv_audio,
            str(tmp_path),
            None,  # model_path (auto-detect)
            threshold,
            sample_rate,
        )

        # Ensure prediction_label is set
        if result.get("success") and result.get("prediction") is not None:
            if int(result["prediction"]) == 1:
                result["prediction_label"] = "experimental"
            else:
                result["prediction_label"] = "control"

        return result

    finally:
        # Cleanup
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        try:
            Path(tmp_dir).rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
