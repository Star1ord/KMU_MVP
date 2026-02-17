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

# Load models with error details
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


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    skip_transcription: bool = False,
    label: int | None = None,
):
    """Upload a video file, run preprocessing + models, and return the analysis result."""

    if analyze_new_media_file is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server missing preprocessing functionality")

    # Validate content type
    allowed = {"video/mp4", "video/mpeg", "video/quicktime", "video/x-matroska", "video/webm"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    tmp_dir = tempfile.mkdtemp(prefix="api_upload_")
    tmp_path = Path(tmp_dir) / file.filename
    try:
        with open(tmp_path, "wb") as out:
            content = await file.read()
            out.write(content)

        # Run processing in threadpool to avoid blocking the event loop
        result = await run_in_threadpool(
            analyze_new_media_file,
            str(tmp_path),
            label,
            False,  # add_to_training
            'medium',  # whisper_model
            skip_transcription,
        )

        return result
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

@app.post("/test/nlp")
def test_nlp(features: dict):
    if 'nlp' not in models: 
        return {"error": "NLP not loaded"}
    try:
        X = np.array([list(features.values())])
        score = models['nlp'].predict_proba(X)[0, 1]
        return {"score": float(score)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/test/cv")
def test_cv(features: dict):
    if 'cv' not in models:
        return {"error": "CV model not loaded"}
    try:
        X = np.array([list(features.values())])
        score = models['cv'].predict_proba(X)[0, 1]
        return {"score": float(score)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
