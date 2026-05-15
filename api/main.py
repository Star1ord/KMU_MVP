from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib, pickle
from pathlib import Path
import numpy as np
import logging
import pandas as pd
from datetime import datetime
from urllib.parse import unquote

from src.utils.model_loading import ModelLoadError, safe_load_serialized_model
from src.utils.paths import CV_AUDIO_MODELS_DIR, CV_MODELS_DIR, NLP_MODELS_DIR

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
    'nlp': NLP_MODELS_DIR / 'early_fusion_linear_svc.pkl',
    'cv': CV_MODELS_DIR / 'model.joblib',
    'cv_audio': CV_AUDIO_MODELS_DIR / 'model.joblib',
}

models = {}
for name, path in MODEL_PATHS.items():
    if path.exists() and path.stat().st_size > 0:
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
        if path.exists():
            logger.warning(f"⚠ {name} model file is empty at {path}")
        else:
            logger.warning(f"⚠ {name} not found at {path}")


safe_models = {}
for name, path in MODEL_PATHS.items():
    if not path.exists() or path.stat().st_size <= 0:
        continue
    try:
        safe_models[name] = safe_load_serialized_model(path, model_name=name)
    except ModelLoadError as exc:
        logger.error("%s safe-load error: %s", name, exc)

if safe_models:
    models = safe_models

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
    allowed_content_types = {
        "video/mp4",
        "application/mp4",
        "application/x-mp4",
        "video/mpeg",
        "video/quicktime",
        "video/x-matroska",
        "application/x-matroska",
        "video/webm",
        "video/x-msvideo",
        "video/avi",
        "video/msvideo",
        "video/3gpp",
        "video/3gpp2",
        "video/x-flv",
        "video/mp2t",
        "application/octet-stream",
    }

    allowed_extensions = {
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mov",
        ".mkv",
        ".webm",
        ".avi",
        ".m4v",
        ".3gp",
        ".3g2",
        ".mts",
        ".m2ts",
        ".ts",
        ".flv",
        ".wmv",
    }

    content_type = (file.content_type or "").strip().lower()
    suffix = Path(file.filename or "").suffix.lower()

    # In practice browser / OS MIME detection is unreliable for local videos.
    # If the filename extension is a known video container, let the pipeline
    # validate the file contents later with ffmpeg/OpenCV instead of rejecting
    # the upload up front with a brittle 400.
    if suffix in allowed_extensions:
        logger.info(
            "Accepted upload by extension: filename=%r content_type=%r",
            file.filename,
            file.content_type,
        )
        return

    if content_type in allowed_content_types:
        return

    if content_type.startswith("video/"):
        return

    logger.warning(
        "Rejected upload: filename=%r content_type=%r",
        file.filename,
        file.content_type,
    )
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"Unsupported file type: content_type={file.content_type!r}, "
            f"filename={file.filename!r}"
        ),
    )


async def _analyze_uploaded_video(
    file: UploadFile,
    skip_transcription: bool = False,
    label: int | None = None,
    include_cv: bool = True,
    include_cv_audio: bool = False,
    whisper_model: str = "medium",
    video_sample_every: int = 8,
    cv_audio_sample_rate: float = 0.25,
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

        logger.info(f"Processing upload: {file.filename}")
        
        # Run processing in threadpool to avoid blocking the event loop
        result = await run_in_threadpool(
            analyze_new_media_file,
            str(tmp_path),
            label,
            False,  # add_to_training
            whisper_model,
            skip_transcription,
            (not include_cv),  # skip_video
            (not include_cv_audio),  # skip_cv_audio
            int(video_sample_every),
            float(cv_audio_sample_rate),
        )
        
        if result is None:
            logger.error(f"analyze_new_media_file returned None for {file.filename}")
            raise ValueError("Processing returned no result")
        
        result_success = result.get("formatted_result", result).get("success")
        logger.info(f"Processing complete for {file.filename}: success={result_success}")
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
    request: Request,
    file: UploadFile | None = File(default=None),
    skip_transcription: bool = False,
    label: int | None = None,
    include_cv: bool = True,
    include_cv_audio: bool = False,
    video_sample_every: int = 8,
    cv_audio_sample_rate: float = 0.25,
):
    """
    End-to-end production endpoint.

    Upload a video file, run full preprocessing + NLP (and optional CV) models,
    and return the formatted result structure.
    """
    if file is not None:
        logger.info("/predict received multipart upload: %s", file.filename)
        result = await _analyze_uploaded_video(
            file=file,
            skip_transcription=skip_transcription,
            label=label,
            include_cv=include_cv,
            include_cv_audio=include_cv_audio,
            whisper_model="medium",
            video_sample_every=video_sample_every,
            cv_audio_sample_rate=cv_audio_sample_rate,
        )
        output = result.get("formatted_result", result)
        logger.info("/predict returned successfully for multipart upload %s", file.filename)
        return output

    tmp_dir = None
    tmp_path = None
    display_name = "upload"
    try:
        display_name, _content_type, tmp_dir, tmp_path = await _save_raw_request_body_to_temp_file(request)
        result = await _analyze_saved_video(
            tmp_path=tmp_path,
            display_name=display_name,
            skip_transcription=skip_transcription,
            label=label,
            include_cv=include_cv,
            include_cv_audio=include_cv_audio,
            whisper_model="medium",
            video_sample_every=video_sample_every,
            cv_audio_sample_rate=cv_audio_sample_rate,
        )
        
        if result is None:
            logger.error("_analyze_uploaded_video returned None")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Processing failed: no result returned"
            )
        
        # Return formatted result if available, otherwise raw
        output = result.get('formatted_result', result)
        logger.info("/predict returned successfully for %s", display_name)
        return output
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ /predict error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        try:
            if tmp_dir is not None:
                Path(tmp_dir).rmdir()
        except Exception:
            pass


@app.post("/test/nlp")
async def test_nlp(
    file: UploadFile = File(...),
    skip_transcription: bool = False,
    label: int | None = None,
):
    """
    NLP-focused endpoint.
    Returns compact NLP result with prediction, risk_score, and risk_level.
    """
    try:
        result = await _analyze_uploaded_video(
            file=file,
            skip_transcription=skip_transcription,
            label=label,
            include_cv=False,
            include_cv_audio=False,
            whisper_model="small",
        )
        
        if result is None:
            logger.error("_analyze_uploaded_video returned None for /test/nlp")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Processing failed: no result returned"
            )
        
        # Extract raw result (contains top-level prediction/risk_score from NLP model)
        raw = result.get('raw_result', result)
        
        if raw is None:
            logger.error("raw_result extraction failed for /test/nlp")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Processing failed: could not extract results"
            )

        prediction = raw.get("prediction")
        risk_score = raw.get("risk_score")
        risk_level = raw.get("risk_level")
        segments = raw.get("segments") or raw.get("text_segments") or []
        transcript = (
            str(raw.get("transcript") or "").strip()
            or str(raw.get("full_text") or "").strip()
            or " ".join(
                str(seg.get("text", "")).strip()
                for seg in segments
                if str(seg.get("text", "")).strip()
            ).strip()
        )

        # 0 = контрольная группа, 1 = экспериментальная (risk)
        if prediction == 1:
            prediction_label = "experimental"  # RISK
        elif prediction == 0:
            prediction_label = "control"  # CONTROL
        else:
            prediction_label = None

        output = {
            "success": bool(raw.get("nlp_success", raw.get("success", False))),
            "prediction": prediction,
            "prediction_label": prediction_label,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "transcript": transcript,
            "full_text": transcript,
            "raw_text": transcript,
            "segments": segments,
        }
        logger.info(f"✓ /test/nlp returned successfully for {file.filename}")
        return output
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ /test/nlp error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@app.post("/test/cv")
async def test_cv(
    file: UploadFile = File(...),
    sample_every: int = 8,
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
            sample_every=max(1, int(sample_every)),
            use_emotions=False,
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
    sample_rate: float = 0.25,
    use_vgg: bool = False,
):
    """
    CV+Audio endpoint: принимает видео, извлекает лица (RetinaFace), 
    VGG16 визуальные фичи + аудио фичи, и прогоняет через CV+Audio модель.
    
    Возвращает probability, prediction, prediction_label и risk_level.
    """
    try:
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
                13,  # n_mfcc
                use_vgg,
            )

            if result is None:
                logger.error("predict_cv_audio returned None")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="CV+Audio processing failed: no result returned"
                )

            # Ensure prediction_label is set
            if result.get("success") and result.get("prediction") is not None:
                if int(result["prediction"]) == 1:
                    result["prediction_label"] = "experimental"
                else:
                    result["prediction_label"] = "control"

            logger.info(f"✓ /predict/cv-audio returned successfully for {file.filename}")
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ /predict/cv-audio error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _risk_level_from_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _prediction_label(prediction: int | None) -> str | None:
    if prediction is None:
        return None
    return "experimental" if int(prediction) == 1 else "control"


def _segment_bounds(seg: dict) -> tuple[float, float, float]:
    start = seg.get("start_time", seg.get("start", 0.0)) or 0.0
    end = seg.get("end_time", seg.get("end", start)) or start
    duration = seg.get("duration")
    if duration is None:
        duration = float(max(0.0, float(end) - float(start)))
    return float(start), float(end), float(duration)


def _cleanup_temp_csv(csv_path: str | None) -> None:
    if not csv_path:
        return
    try:
        p = Path(csv_path)
        if p.exists():
            p.unlink()
        parent = p.parent
        if parent.name.startswith("video_features_"):
            try:
                parent.rmdir()
            except Exception:
                pass
    except Exception:
        pass


@app.post("/test/deception")
async def test_deception(
    file: UploadFile = File(...),
    skip_transcription: bool = False,
    label: int | None = None,
):
    """
    Deception-style proxy model based on segment-level risk dynamics.
    Uses only NLP pipeline outputs (no CV/CV+Audio).
    """
    try:
        result = await _analyze_uploaded_video(
            file=file,
            skip_transcription=skip_transcription,
            label=label,
            include_cv=False,
            include_cv_audio=False,
            whisper_model="small",
        )
        raw = result.get("raw_result", result)
        segments = raw.get("segments", []) if isinstance(raw, dict) else []

        series = []
        for seg in segments:
            score = seg.get("risk_score")
            if score is None:
                continue
            start, end, duration = _segment_bounds(seg)
            if 3.0 <= duration <= 13.0:
                series.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "score": float(score),
                    }
                )

        if not series:
            for seg in segments:
                score = seg.get("risk_score")
                if score is None:
                    continue
                start, end, duration = _segment_bounds(seg)
                series.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "score": float(score),
                    }
                )

        if not series:
            return {
                "success": False,
                "deception_score": None,
                "prediction": None,
                "prediction_label": None,
                "risk_level": None,
                "segments_used": 0,
                "time_series": [],
                "error": "No segment-level scores available for deception analysis",
            }

        scores = np.array([x["score"] for x in series], dtype=float)
        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
        high_ratio = float(np.mean(scores >= 0.6))

        deception_score = _clamp01(0.50 * score_mean + 0.35 * score_std + 0.15 * high_ratio)
        prediction = 1 if deception_score >= 0.55 else 0

        return {
            "success": True,
            "deception_score": deception_score,
            "prediction": prediction,
            "prediction_label": _prediction_label(prediction),
            "risk_level": _risk_level_from_score(deception_score),
            "segments_used": len(series),
            "time_series": series,
            "components": {
                "mean": score_mean,
                "std": score_std,
                "high_ratio": high_ratio,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ /test/deception error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@app.post("/test/emotion-av")
async def test_emotion_av(
    file: UploadFile = File(...),
    sample_every: int = 8,
    sample_rate: float = 0.25,
):
    """
    Emotion audio+video proxy:
    - video emotion trajectory from CV pipeline CSV
    - optional CV+Audio probability as arousal proxy
    """
    if process_video_for_prediction is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video CV pipeline not available on this server",
        )

    _validate_video_content_type(file)
    tmp_dir = tempfile.mkdtemp(prefix="api_emotion_upload_")
    tmp_path = Path(tmp_dir) / file.filename
    csv_path = None

    try:
        with open(tmp_path, "wb") as out:
            content = await file.read()
            out.write(content)

        csv_path, video_err = process_video_for_prediction(
            video_path=str(tmp_path),
            output_dir=None,
            sample_every=max(1, int(sample_every)),
            use_emotions=True,
        )
        if video_err or not csv_path:
            return {
                "success": False,
                "error": video_err or "Failed to compute video emotion features",
            }

        df = pd.read_csv(csv_path)
        if "emotion" in df.columns:
            emo = (
                df["emotion"]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .str.lower()
            )
            emo = emo[emo != "unknown"]
        else:
            emo = pd.Series([], dtype=str)

        distribution = {}
        dominant_emotion = "unknown"
        negative_ratio = 0.0
        if len(emo) > 0:
            counts = emo.value_counts(normalize=True)
            distribution = {k: float(v) for k, v in counts.to_dict().items()}
            dominant_emotion = str(counts.index[0])
            negative_ratio = float(
                distribution.get("sad", 0.0)
                + distribution.get("angry", 0.0)
                + distribution.get("fear", 0.0)
                + distribution.get("disgust", 0.0)
                + distribution.get("contempt", 0.0)
            )

        cv_audio_prob = None
        cv_audio_ok = False
        cv_audio_error = None
        if predict_cv_audio is not None:
            cv_audio_result = await run_in_threadpool(
                predict_cv_audio,
                str(tmp_path),
                None,
                0.5,
                max(0.1, float(sample_rate)),
            )
            if isinstance(cv_audio_result, dict) and cv_audio_result.get("success"):
                cv_audio_ok = True
                cv_audio_prob = cv_audio_result.get("probability")
            elif isinstance(cv_audio_result, dict):
                cv_audio_error = cv_audio_result.get("error")

        emotion_score = _clamp01(
            0.70 * negative_ratio + 0.30 * float(cv_audio_prob if cv_audio_prob is not None else 0.0)
        )
        prediction = 1 if emotion_score >= 0.5 else 0

        return {
            "success": True,
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": distribution,
            "negative_ratio": negative_ratio,
            "cv_audio_probability": float(cv_audio_prob) if cv_audio_prob is not None else None,
            "cv_audio_success": cv_audio_ok,
            "cv_audio_error": cv_audio_error,
            "emotion_score": emotion_score,
            "prediction": prediction,
            "prediction_label": _prediction_label(prediction),
            "risk_level": _risk_level_from_score(emotion_score),
        }
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        try:
            Path(tmp_dir).rmdir()
        except Exception:
            pass
        _cleanup_temp_csv(csv_path)


def _normalize_upload_name(filename: str | None, default_name: str = "upload.mp4") -> str:
    candidate = (filename or "").strip()
    if not candidate:
        return default_name

    candidate = candidate.replace("\\", "/").split("/")[-1].strip()
    return candidate or default_name


async def _save_raw_request_body_to_temp_file(request: Request) -> tuple[str, str, str, Path]:
    header_name = request.headers.get("x-file-name")
    content_type = (
        request.headers.get("x-file-content-type")
        or request.headers.get("content-type")
        or "application/octet-stream"
    ).split(";", 1)[0].strip().lower()
    default_suffix = {
        "video/mp4": ".mp4",
        "application/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-matroska": ".mkv",
        "video/webm": ".webm",
        "video/x-msvideo": ".avi",
    }.get(content_type, ".mp4")
    default_name = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{default_suffix}"
    filename = _normalize_upload_name(unquote(header_name) if header_name else None, default_name=default_name)

    file_meta = type(
        "RequestFileMeta",
        (),
        {"filename": filename, "content_type": content_type},
    )()
    _validate_video_content_type(file_meta)

    body = await request.body()
    if not body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload body",
        )

    tmp_dir = tempfile.mkdtemp(prefix="api_raw_upload_")
    tmp_path = Path(tmp_dir) / filename
    with open(tmp_path, "wb") as out:
        out.write(body)

    logger.info(
        "Saved raw upload: filename=%r content_type=%r bytes=%s",
        filename,
        content_type,
        len(body),
    )
    return filename, content_type, tmp_dir, tmp_path


async def _analyze_saved_video(
    tmp_path: Path,
    display_name: str,
    *,
    skip_transcription: bool = False,
    label: int | None = None,
    include_cv: bool = True,
    include_cv_audio: bool = False,
    whisper_model: str = "medium",
    video_sample_every: int = 8,
    cv_audio_sample_rate: float = 0.25,
) -> dict:
    if analyze_new_media_file is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server missing preprocessing functionality",
        )

    logger.info("Processing saved upload: %s", display_name)

    result = await run_in_threadpool(
        analyze_new_media_file,
        str(tmp_path),
        label,
        False,
        whisper_model,
        skip_transcription,
        (not include_cv),
        (not include_cv_audio),
        int(video_sample_every),
        float(cv_audio_sample_rate),
    )

    if result is None:
        logger.error("analyze_new_media_file returned None for %s", display_name)
        raise ValueError("Processing returned no result")

    result_success = result.get("formatted_result", result).get("success")
    logger.info("Processing complete for %s: success=%s", display_name, result_success)
    return result


@app.post("/test/anomaly/text")
async def test_anomaly_text(
    file: UploadFile = File(...),
    skip_transcription: bool = False,
    label: int | None = None,
):
    try:
        result = await _analyze_uploaded_video(
            file=file,
            skip_transcription=skip_transcription,
            label=label,
            include_cv=False,
            include_cv_audio=False,
            whisper_model="small",
        )
        raw = result.get("raw_result", result)
        segments = raw.get("segments", []) if isinstance(raw, dict) else []
        texts = [str(s.get("text", "")).strip() for s in segments if str(s.get("text", "")).strip()]
        full_text = " ".join(texts)

        import re

        tokens = re.findall(r"\w+", full_text.lower())
        word_count = len(tokens)
        unique_count = len(set(tokens))
        lexical_diversity = (unique_count / word_count) if word_count > 0 else 0.0
        repetition_ratio = 1.0 - lexical_diversity if word_count > 0 else 1.0
        punct_count = len(re.findall(r"[!?.,;:]", full_text))
        punct_ratio = punct_count / max(1, len(full_text))
        short_text_flag = 1.0 if word_count < 25 else 0.0

        anomaly_score = _clamp01(
            0.55 * repetition_ratio + 0.25 * short_text_flag + 0.20 * min(1.0, punct_ratio * 4.0)
        )
        prediction = 1 if anomaly_score >= 0.6 else 0

        return {
            "success": True,
            "modality": "text",
            "anomaly_score": anomaly_score,
            "prediction": prediction,
            "prediction_label": "anomaly" if prediction == 1 else "normal",
            "risk_level": _risk_level_from_score(anomaly_score),
            "metrics": {
                "word_count": word_count,
                "unique_words": unique_count,
                "lexical_diversity": lexical_diversity,
                "repetition_ratio": repetition_ratio,
                "punctuation_ratio": punct_ratio,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ /test/anomaly/text error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@app.post("/test/anomaly/audio")
async def test_anomaly_audio(
    file: UploadFile = File(...),
    skip_transcription: bool = False,
    label: int | None = None,
):
    try:
        result = await _analyze_uploaded_video(
            file=file,
            skip_transcription=skip_transcription,
            label=label,
            include_cv=False,
            include_cv_audio=False,
            whisper_model="small",
        )
        raw = result.get("raw_result", result)
        segments = raw.get("segments", []) if isinstance(raw, dict) else []

        scores = []
        durations = []
        time_series = []
        for seg in segments:
            score = seg.get("risk_score")
            if score is None:
                continue
            start, end, duration = _segment_bounds(seg)
            scores.append(float(score))
            durations.append(float(duration))
            time_series.append({"start": start, "end": end, "duration": duration, "score": float(score)})

        if not scores:
            return {
                "success": False,
                "modality": "audio",
                "anomaly_score": None,
                "prediction": None,
                "prediction_label": None,
                "risk_level": None,
                "time_series": [],
                "error": "No audio segment scores available",
            }

        arr_scores = np.array(scores, dtype=float)
        arr_dur = np.array(durations, dtype=float) if durations else np.array([0.0], dtype=float)
        score_std = float(np.std(arr_scores)) if len(arr_scores) > 1 else 0.0
        duration_cv = float(np.std(arr_dur) / max(1e-6, np.mean(arr_dur)))
        jump_ratio = (
            float(np.mean(np.abs(np.diff(arr_scores)) > 0.35)) if len(arr_scores) > 1 else 0.0
        )
        low_segments_flag = 1.0 if len(arr_scores) < 3 else 0.0

        anomaly_score = _clamp01(
            0.40 * score_std + 0.30 * min(1.0, duration_cv) + 0.20 * jump_ratio + 0.10 * low_segments_flag
        )
        prediction = 1 if anomaly_score >= 0.6 else 0

        return {
            "success": True,
            "modality": "audio",
            "anomaly_score": anomaly_score,
            "prediction": prediction,
            "prediction_label": "anomaly" if prediction == 1 else "normal",
            "risk_level": _risk_level_from_score(anomaly_score),
            "time_series": time_series,
            "metrics": {
                "segments_used": len(arr_scores),
                "score_std": score_std,
                "duration_cv": duration_cv,
                "jump_ratio": jump_ratio,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ /test/anomaly/audio error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@app.post("/test/anomaly/video")
async def test_anomaly_video(
    file: UploadFile = File(...),
    sample_every: int = 8,
):
    if process_video_for_prediction is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video CV pipeline not available on this server",
        )

    _validate_video_content_type(file)
    tmp_dir = tempfile.mkdtemp(prefix="api_anom_video_upload_")
    tmp_path = Path(tmp_dir) / file.filename
    csv_path = None

    try:
        with open(tmp_path, "wb") as out:
            content = await file.read()
            out.write(content)

        csv_path, video_err = process_video_for_prediction(
            video_path=str(tmp_path),
            output_dir=None,
            sample_every=max(1, int(sample_every)),
            use_emotions=False,
        )
        if video_err or not csv_path:
            return {
                "success": True,
                "modality": "video",
                "anomaly_score": 1.0,
                "prediction": 1,
                "prediction_label": "anomaly",
                "risk_level": "high",
                "reason": video_err or "Video features were not produced",
            }

        df = pd.read_csv(csv_path)
        indicators = {}

        def col_max_gt(col: str, threshold: float) -> float:
            if col not in df.columns:
                return 0.0
            return float(df[col].fillna(0.0).max() > threshold)

        def col_mean_lt(col: str, threshold: float) -> float:
            if col not in df.columns:
                return 0.0
            return float(df[col].fillna(0.0).mean() < threshold)

        indicators["gaze_avoidance"] = col_max_gt("gaze_away_time", 5.0)
        indicators["head_down_long"] = col_max_gt("head_down_duration", 3.0)
        indicators["instability"] = col_mean_lt("head_stability", 0.5)
        indicators["face_touching"] = col_max_gt("face_touch_count", 3.0)
        indicators["blinking_spikes"] = col_max_gt("blink_count", 20.0)

        rule_score = float(np.mean(list(indicators.values()))) if indicators else 0.0

        cv_prob = None
        if predict_with_video_model is not None:
            cv_result = predict_with_video_model(video_csv_path=csv_path, model_path=None, threshold=0.6)
            if cv_result.get("success") and cv_result.get("probability") is not None:
                cv_prob = float(cv_result["probability"])

        anomaly_score = _clamp01(0.70 * rule_score + 0.30 * float(cv_prob if cv_prob is not None else 0.0))
        prediction = 1 if anomaly_score >= 0.6 else 0

        return {
            "success": True,
            "modality": "video",
            "anomaly_score": anomaly_score,
            "prediction": prediction,
            "prediction_label": "anomaly" if prediction == 1 else "normal",
            "risk_level": _risk_level_from_score(anomaly_score),
            "cv_probability": cv_prob,
            "indicators": indicators,
        }
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        try:
            Path(tmp_dir).rmdir()
        except Exception:
            pass
        _cleanup_temp_csv(csv_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
