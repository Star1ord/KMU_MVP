"""
CV+Audio inference (video -> MFCC + VGG16 -> ONNX DNN).

Based on the training notebook `cv_audio_test/audio_cv_1to1.py`:
- audio features: MFCC (13 dims)
- video features: VGG16 fc1+fc2 (8192 dims) averaged over frames
- input order during training: [audio, video] -> 8205 dims
- both halves are StandardScaler-normalised before the net (scaler_audio: 13,
  scaler_video: 8192); the ONNX outputs raw logits, class 1 = experimental.
"""

from __future__ import annotations

import math
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


_ONNX_SESSION = None
_ONNX_SESSION_PATH: Optional[str] = None

_VGG_MODELS: Dict[str, Any] = {}
_HAAR_CASCADE = None
_RETINAFACE_AVAILABLE: Optional[bool] = None
_RETINAFACE_CLASS = None

_SCALER_AUDIO = None
_SCALER_VIDEO = None
_SCALERS_LOADED = False

# StandardScaler divides by per-dim std; a few VGG fc dims are near-constant in the
# training set (min std ~3e-5), so a tiny live deviation explodes to |z|~1e4 and
# saturates the net. Training-scaled features sit within ~[-5,5], so clipping at this
# bound is a no-op in-distribution but tames pathological out-of-distribution dims.
_SCALED_CLIP = 10.0

_VGG_UNAVAILABLE_MSG = (
    "VGG16 visual features unavailable: TensorFlow/Keras is not installed in this "
    "runtime. The CV+Audio ONNX was trained on Keras VGG16 fc1+fc2 features, so the "
    "grayscale+Sobel fallback is not interchangeable. Install tensorflow-cpu (and let "
    "VGG16 imagenet weights download) to enable this modality, or pass allow_fallback=True "
    "to accept an unreliable descriptor."
)


def _safe_metric_value(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric) or np.isinf(numeric):
        return None
    return numeric


def _load_cv_audio_metrics() -> Dict[str, Optional[float]]:
    """Read public metrics from optional CV+Audio metadata, preferring recall."""
    metadata_path = Path("models/cv_audio/metadata.json")
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "recall": None,
            "f1_score": None,
        }

    try:
        with metadata_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "recall": None,
            "f1_score": None,
        }

    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    model_metrics = payload.get("model_metrics", {}) if isinstance(payload, dict) else {}

    recall = _safe_metric_value(payload.get("mean_recall"))
    if recall is None:
        recall = _safe_metric_value(payload.get("recall_mean"))
    if recall is None:
        recall = _safe_metric_value(metrics.get("recall"))
    if recall is None:
        recall = _safe_metric_value(model_metrics.get("recall"))
    if recall is None:
        recall = _safe_metric_value(payload.get("test_metrics", {}).get("recall"))

    roc_auc = _safe_metric_value(payload.get("mean_roc_auc"))
    if roc_auc is None:
        roc_auc = _safe_metric_value(metrics.get("roc_auc"))

    pr_auc = _safe_metric_value(payload.get("mean_pr_auc"))
    if pr_auc is None:
        pr_auc = _safe_metric_value(metrics.get("pr_auc"))

    f1_score = _safe_metric_value(payload.get("mean_f1"))
    if f1_score is None:
        f1_score = _safe_metric_value(metrics.get("f1"))
    if f1_score is None:
        f1_score = _safe_metric_value(model_metrics.get("f1_score"))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": recall,
        "f1_score": f1_score,
    }


def _find_onnx_model_path(model_path: Optional[str] = None) -> str:
    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"CV+Audio ONNX model not found: {p}")
        return str(p)

    candidates = [
        Path("models/cv_audio/DNN_model.onnx"),
        Path("models/cv_audio/dnn_model.onnx"),
        Path("cv_audio_test/DNN_model.onnx"),
        Path("cv_audio_test/dnn_model.onnx"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "CV+Audio ONNX model not found. Put it in `models/cv_audio/DNN_model.onnx` "
        f"or provide explicit model_path. Checked: {[str(c) for c in candidates]}"
    )


def _get_onnx_session(model_path: str):
    global _ONNX_SESSION, _ONNX_SESSION_PATH
    if _ONNX_SESSION is not None and _ONNX_SESSION_PATH == model_path:
        return _ONNX_SESSION

    try:
        import onnxruntime as ort
    except Exception as e:
        raise ImportError(
            "onnxruntime is required for CV+Audio ONNX inference. "
            "Install: pip install onnxruntime"
        ) from e

    # Prefer CPU to keep deployment simple
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _ONNX_SESSION = sess
    _ONNX_SESSION_PATH = model_path
    return sess


def _load_scalers() -> Tuple[Any, Any]:
    """Load the StandardScalers used at training time (audio: 13 dims, video: 8192 dims).

    These are mandatory: the ONNX was trained on features normalised by these exact
    scalers. Feeding raw (unscaled) features collapses the output to ~0 regardless of
    the actual content. Returns (scaler_audio, scaler_video) or (None, None) if missing.
    """
    global _SCALER_AUDIO, _SCALER_VIDEO, _SCALERS_LOADED
    if _SCALERS_LOADED:
        return _SCALER_AUDIO, _SCALER_VIDEO

    _SCALERS_LOADED = True
    try:
        import joblib
    except Exception:
        return None, None

    audio_candidates = [
        Path("models/cv_audio/scaler_audio.pkl"),
        Path("cv_audio_test/scaler_audio.pkl"),
    ]
    video_candidates = [
        Path("models/cv_audio/scaler_video.pkl"),
        Path("cv_audio_test/scaler_video.pkl"),
    ]

    def _first_existing(paths):
        for p in paths:
            if p.exists():
                return p
        return None

    audio_path = _first_existing(audio_candidates)
    video_path = _first_existing(video_candidates)
    if audio_path is None or video_path is None:
        return None, None

    try:
        _SCALER_AUDIO = joblib.load(audio_path)
        _SCALER_VIDEO = joblib.load(video_path)
    except Exception:
        _SCALER_AUDIO = None
        _SCALER_VIDEO = None

    return _SCALER_AUDIO, _SCALER_VIDEO


def _ensure_haar():
    global _HAAR_CASCADE
    if _HAAR_CASCADE is not None:
        return _HAAR_CASCADE

    import cv2

    haar_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(haar_path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {haar_path}")
    _HAAR_CASCADE = cascade
    return cascade


def _read_video_rotation(video_path: str) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return 0

    values: list[int] = []
    for line in completed.stdout.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            values.append(int(round(float(candidate))))
        except ValueError:
            continue

    if not values:
        return 0

    return values[-1] % 360


def _apply_rotation(frame: np.ndarray, rotation_degrees: int) -> np.ndarray:
    import cv2

    rotation = rotation_degrees % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def _extract_visual_crop(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    bbox = _detect_face_bbox_bgr(frame_bgr)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], x2)
        y2 = min(frame_bgr.shape[0], y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0 and crop.shape[0] >= 50 and crop.shape[1] >= 50:
            return crop

    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    if side < 50:
        return None

    side = max(50, int(side * 0.82))
    side = min(side, height, width)
    x1 = max(0, (width - side) // 2)
    y1 = max(0, (height - side) // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _extract_audio_mfcc(
    video_path: str,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> np.ndarray:
    """
    Extract MFCC mean vector from video's audio track.
    Returns shape: (n_mfcc,).
    """
    try:
        import soundfile as sf
        import torch
        import torchaudio
    except Exception as e:
        raise ImportError(
            "soundfile and torchaudio are required for MFCC extraction. "
            "Install: pip install soundfile torchaudio"
        ) from e

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        # Convert to mono wav @ sr
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            wav_path,
        ]
        subprocess.run(cmd, check=True)

        waveform, loaded_sr = sf.read(wav_path, always_2d=False)
        if waveform is None or len(waveform) == 0:
            raise ValueError("empty audio after ffmpeg extraction")

        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)

        waveform = waveform.astype(np.float32)
        if waveform.size == 0:
            raise ValueError("empty waveform after mono conversion")

        tensor = torch.from_numpy(waveform).unsqueeze(0)
        transform = torchaudio.transforms.MFCC(
            sample_rate=int(loaded_sr),
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40,
                "center": False,
            },
        )
        mfcc = transform(tensor)
        if mfcc.numel() == 0:
            raise ValueError("mfcc extraction produced empty output")

        return (
            mfcc.mean(dim=-1)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def _get_vgg16_fc_models():
    """
    Load VGG16 and expose fc1 + fc2 layers as feature extractors.
    Cached globally to avoid re-loading per request.
    """
    if "fc1" in _VGG_MODELS and "fc2" in _VGG_MODELS:
        return _VGG_MODELS["fc1"], _VGG_MODELS["fc2"]

    try:
        _ensure_protobuf_runtime_compat()
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Model
    except Exception as e:
        raise ImportError(
            "TensorFlow/Keras is required for VGG16 feature extraction. "
            "Install a compatible TensorFlow build for your Python version."
        ) from e

    base = VGG16(weights="imagenet", include_top=True)
    fc1 = Model(inputs=base.input, outputs=base.get_layer("fc1").output)
    fc2 = Model(inputs=base.input, outputs=base.get_layer("fc2").output)
    _VGG_MODELS["fc1"] = fc1
    _VGG_MODELS["fc2"] = fc2
    return fc1, fc2


def _get_vgg16_feat_model():
    """VGG16 model emitting [fc1, fc2] in one forward pass (one conv pass, not two).

    Numerically identical to concatenating the separate fc1/fc2 models used at
    training time, but ~2x cheaper per frame. Cached globally.
    """
    if "feat" in _VGG_MODELS:
        return _VGG_MODELS["feat"]

    _ensure_protobuf_runtime_compat()
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Model

    base = VGG16(weights="imagenet", include_top=True)
    feat = Model(
        inputs=base.input,
        outputs=[base.get_layer("fc1").output, base.get_layer("fc2").output],
    )
    _VGG_MODELS["feat"] = feat
    return feat


def _load_audio_waveform(video_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Decode the full audio track to a mono float32 waveform at `sr`."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr), wav_path,
        ]
        subprocess.run(cmd, check=True)
        waveform, loaded_sr = sf.read(wav_path, always_2d=False)
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        return np.asarray(waveform, dtype=np.float32), int(loaded_sr)
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def _mfcc_mean(waveform: np.ndarray, sr: int, n_mfcc: int = 13) -> Optional[np.ndarray]:
    """Mean MFCC vector over a waveform slice. Returns (n_mfcc,) or None if empty."""
    import torch
    import torchaudio

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        return None
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    transform = torchaudio.transforms.MFCC(
        sample_rate=int(sr),
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40, "center": False},
    )
    mfcc = transform(tensor)
    if mfcc.numel() == 0:
        return None
    return mfcc.mean(dim=-1).squeeze(0).detach().cpu().numpy().astype(np.float32)


def _ensure_protobuf_runtime_compat() -> None:
    """
    Best-effort compatibility shim for environments where protobuf is older
    than code generated by newer protoc versions.
    """
    try:
        import google.protobuf as _pb  # type: ignore
    except Exception:
        return

    if hasattr(_pb, "runtime_version"):
        return

    class _RuntimeVersionShim:
        class Domain:
            PUBLIC = 0
            INTERNAL = 1

        @staticmethod
        def ValidateProtobufRuntimeVersion(*args, **kwargs):
            return None

    _pb.runtime_version = _RuntimeVersionShim()  # type: ignore[attr-defined]


def _extract_video_features_fallback(
    video_path: str,
    sample_rate_fps: float = 1.0,
    max_frames: int = 48,
) -> np.ndarray:
    """
    TensorFlow-free visual descriptor fallback (8192 dims):
    - 4096 dims: grayscale 64x64 face crop
    - 4096 dims: Sobel magnitude 64x64 face crop
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    rotation_degrees = _read_video_rotation(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(math.floor(fps / max(sample_rate_fps, 1e-6))))

    feats = []
    frame_idx = 0
    processed = 0

    try:
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % interval == 0:
                frame = _apply_rotation(frame, rotation_degrees)
                crop = _extract_visual_crop(frame)
                if crop is not None:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
                    gray_f = gray.astype(np.float32) / 255.0

                    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
                    mag = cv2.magnitude(gx, gy)
                    mag = np.clip(mag, 0.0, 1.0)

                    f = np.concatenate([gray_f.reshape(-1), mag.reshape(-1)], axis=0)
                    feats.append(f.astype(np.float32))
                    processed += 1

            frame_idx += 1
    finally:
        cap.release()

    if not feats:
        raise ValueError("no face features extracted (fallback mode)")

    feat = np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32)
    if feat.shape[0] != 8192:
        raise ValueError(f"fallback feature vector has wrong shape: {feat.shape}")
    return feat


def _detect_face_bbox_bgr(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Return (x1, y1, x2, y2) for the first detected face.
    Uses RetinaFace if installed; otherwise Haar cascade.
    """
    # 1) RetinaFace if available (imported once, then cached)
    global _RETINAFACE_AVAILABLE, _RETINAFACE_CLASS
    if _RETINAFACE_AVAILABLE is None:
        try:
            from retinaface import RetinaFace  # type: ignore
            _RETINAFACE_CLASS = RetinaFace
            _RETINAFACE_AVAILABLE = True
        except Exception:
            _RETINAFACE_AVAILABLE = False
            _RETINAFACE_CLASS = None

    if _RETINAFACE_AVAILABLE and _RETINAFACE_CLASS is not None:
        try:
            detected = _RETINAFACE_CLASS.detect_faces(frame_bgr)
            if isinstance(detected, dict) and detected:
                first_key = next(iter(detected.keys()))
                x1, y1, x2, y2 = map(int, detected[first_key]["facial_area"])
                return x1, y1, x2, y2
        except Exception:
            pass

    # 2) Haar fallback
    import cv2

    cascade = _ensure_haar()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return int(x), int(y), int(x + w), int(y + h)


def _extract_video_vgg_features(
    video_path: str,
    sample_rate_fps: float = 1.0,
    max_frames: int = 48,
    use_vgg: bool = True,
    allow_fallback: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    Extract mean VGG16 (fc1+fc2) feature vector from sampled face crops in the video.
    Returns (features(8192,), used_vgg).

    used_vgg=True  -> genuine Keras VGG16 fc1+fc2 features (match the trained scaler/ONNX).
    used_vgg=False -> TF-free grayscale+Sobel descriptor. This does NOT match the training
    distribution and must NOT be fed to the ONNX as if it were VGG. It is produced only when
    allow_fallback=True (explicit opt-in); otherwise this function raises.
    """
    import cv2

    if not use_vgg:
        if allow_fallback:
            return (
                _extract_video_features_fallback(
                    video_path=video_path,
                    sample_rate_fps=sample_rate_fps,
                    max_frames=max_frames,
                ),
                False,
            )
        raise RuntimeError(_VGG_UNAVAILABLE_MSG)

    # Try VGG16; only fall back if explicitly allowed.
    try:
        _ensure_protobuf_runtime_compat()
        from tensorflow.keras.applications.vgg16 import preprocess_input
        fc1, fc2 = _get_vgg16_fc_models()
    except Exception as exc:
        if allow_fallback:
            return (
                _extract_video_features_fallback(
                    video_path=video_path,
                    sample_rate_fps=sample_rate_fps,
                    max_frames=max_frames,
                ),
                False,
            )
        raise RuntimeError(_VGG_UNAVAILABLE_MSG) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    rotation_degrees = _read_video_rotation(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(math.floor(fps / max(sample_rate_fps, 1e-6))))

    feats = []
    frame_idx = 0
    processed = 0

    try:
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % interval == 0:
                frame = _apply_rotation(frame, rotation_degrees)
                crop = _extract_visual_crop(frame)
                if crop is not None:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                    x = np.expand_dims(rgb.astype(np.float32), axis=0)
                    x = preprocess_input(x)

                    f1 = fc1.predict(x, verbose=0)
                    f2 = fc2.predict(x, verbose=0)
                    f = np.concatenate([f1, f2], axis=1).reshape(-1)
                    feats.append(f.astype(np.float32))
                    processed += 1

            frame_idx += 1
    finally:
        cap.release()

    if not feats:
        raise ValueError("no face features extracted (no faces detected or all crops invalid)")

    return np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32), True


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _onnx_predict_proba(session, x_vec: np.ndarray) -> float:
    """
    Return probability of class=1.
    Handles common ONNX output formats:
    - logits shape (1,2)
    - proba shape (1,2)
    - score shape (1,1) (logit or probability)
    """
    inp = session.get_inputs()[0]
    inp_name = inp.name

    x = x_vec.astype(np.float32).reshape(1, -1)
    outputs = session.run(None, {inp_name: x})
    if not outputs:
        raise RuntimeError("ONNX session returned no outputs")

    out = outputs[0]
    out = np.array(out)

    # (1,2) -> assume logits or probabilities
    if out.ndim == 2 and out.shape[1] == 2:
        # If values look like probabilities already (sum close to 1), take second column
        row_sum = float(out[0].sum())
        if 0.98 <= row_sum <= 1.02 and np.all(out[0] >= 0) and np.all(out[0] <= 1):
            return float(out[0, 1])
        probs = _softmax(out, axis=1)
        return float(probs[0, 1])

    # (1,1) -> sigmoid if outside [0,1]
    if out.ndim == 2 and out.shape[1] == 1:
        v = float(out[0, 0])
        if 0.0 <= v <= 1.0:
            return v
        return float(_sigmoid(np.array([v], dtype=np.float32))[0])

    # (1,) -> treat similarly
    if out.ndim == 1 and out.shape[0] == 1:
        v = float(out[0])
        if 0.0 <= v <= 1.0:
            return v
        return float(_sigmoid(np.array([v], dtype=np.float32))[0])

    raise ValueError(f"Unsupported ONNX output shape: {out.shape}")


def _collect_chunk_face_crops(
    video_path: str,
    chunk_sec: float,
    keep_chunks: set,
    frames_per_chunk: int,
    faces_fps: float,
) -> Dict[int, list]:
    """Single video pass: per kept chunk, up to `frames_per_chunk` RGB 224x224 face crops.

    Mirrors training: faces are sampled ~1 fps (FaceDetection used interval=fps/1) and
    grouped by the 15s chunk they fall into. Returns {chunk_idx: [rgb(224,224,3), ...]}.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    rotation = _read_video_rotation(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(math.floor(fps / max(faces_fps, 1e-6))))

    crops: Dict[int, list] = {}
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % interval == 0:
                chunk = int((frame_idx / fps) // chunk_sec) if fps else 0
                if chunk in keep_chunks and len(crops.get(chunk, [])) < frames_per_chunk:
                    rotated = _apply_rotation(frame, rotation)
                    crop = _extract_visual_crop(rotated)
                    if crop is not None:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                        crops.setdefault(chunk, []).append(rgb.astype(np.float32))
            frame_idx += 1
    finally:
        cap.release()
    return crops


def predict_cv_audio(
    video_path: str,
    model_path: Optional[str] = None,
    threshold: float = 0.5,
    sample_rate: float = 0.25,  # kept for API compatibility; chunking uses faces_fps
    n_mfcc: int = 13,
    use_vgg: bool = True,
    allow_fallback: bool = False,
    apply_scalers: bool = True,
    chunk_sec: float = 15.0,
    frames_per_chunk: int = 6,
    max_chunks: int = 24,
    faces_fps: float = 1.0,
) -> Dict[str, Any]:
    """
    End-to-end CV+Audio prediction for a single video, matching the training pipeline:
    the clip is split into `chunk_sec` windows; each chunk gets MFCC(13) over its audio
    and mean VGG16 fc1+fc2(8192) over its face frames; both are StandardScaler-normalised
    and scored by the ONNX; the session probability is the mean of per-chunk probabilities.

    Long clips are subsampled to `max_chunks` evenly-spaced windows to bound CPU cost.

    Returns {success, probability, prediction, prediction_label, risk_level,
             feature_source, chunks_used, positive_chunk_ratio, chunk_timeline, error}.
    """
    result: Dict[str, Any] = {
        "success": False,
        "probability": None,
        "prediction": None,
        "prediction_label": None,
        "risk_level": None,
        "metrics": _load_cv_audio_metrics(),
        "error": None,
    }

    try:
        sess = _get_onnx_session(_find_onnx_model_path(model_path))
        inp_name = sess.get_inputs()[0].name

        scaler_audio = scaler_video = None
        if apply_scalers:
            scaler_audio, scaler_video = _load_scalers()
            if scaler_audio is None or scaler_video is None:
                raise RuntimeError(
                    "CV+Audio scalers not found. Expected models/cv_audio/scaler_audio.pkl "
                    "(13 dims) and models/cv_audio/scaler_video.pkl (8192 dims). Without them "
                    "the ONNX receives unscaled features and collapses to ~0."
                )

        # VGG feature model (real features required; no silent fallback).
        used_vgg = bool(use_vgg)
        feat_model = None
        preprocess_input = None
        if used_vgg:
            try:
                _ensure_protobuf_runtime_compat()
                from tensorflow.keras.applications.vgg16 import preprocess_input as _pp
                preprocess_input = _pp
                feat_model = _get_vgg16_feat_model()
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError(_VGG_UNAVAILABLE_MSG) from exc
                used_vgg = False
        if not used_vgg and not allow_fallback:
            raise RuntimeError(_VGG_UNAVAILABLE_MSG)

        # Audio + chunk layout.
        waveform, sr = _load_audio_waveform(video_path, sr=16000)
        duration = (len(waveform) / float(sr)) if sr else 0.0
        if duration <= 0:
            raise ValueError("could not determine media duration / empty audio")

        n_chunks = max(1, int(math.ceil(duration / chunk_sec)))
        if n_chunks > max_chunks:
            keep = {int(round(i)) for i in np.linspace(0, n_chunks - 1, max_chunks)}
        else:
            keep = set(range(n_chunks))

        crops_by_chunk = (
            _collect_chunk_face_crops(video_path, chunk_sec, keep, frames_per_chunk, faces_fps)
            if used_vgg else {}
        )

        chunk_probs: list = []
        timeline: list = []
        for chunk in sorted(keep):
            start_s = chunk * chunk_sec
            end_s = min((chunk + 1) * chunk_sec, duration)
            if end_s <= start_s:
                continue

            crops = crops_by_chunk.get(chunk, [])
            if not crops:
                continue  # no face in this window -> skip (graceful, like training)
            batch = preprocess_input(np.stack(crops, axis=0).astype(np.float32))
            f1, f2 = feat_model.predict(batch, verbose=0)
            video_feat = np.concatenate([f1, f2], axis=1).mean(axis=0).astype(np.float32)

            audio_feat = _mfcc_mean(waveform[int(start_s * sr):int(end_s * sr)], sr, n_mfcc=n_mfcc)
            if audio_feat is None or audio_feat.shape[0] != n_mfcc:
                continue

            if apply_scalers:
                audio_feat = np.clip(
                    scaler_audio.transform(audio_feat.reshape(1, -1))[0], -_SCALED_CLIP, _SCALED_CLIP
                ).astype(np.float32)
                video_feat = np.clip(
                    scaler_video.transform(video_feat.reshape(1, -1))[0], -_SCALED_CLIP, _SCALED_CLIP
                ).astype(np.float32)

            x_vec = np.concatenate([audio_feat, video_feat], axis=0).astype(np.float32)
            p = _onnx_predict_proba(sess, x_vec)
            chunk_probs.append(p)
            timeline.append({"start": round(start_s, 2), "end": round(end_s, 2), "prob": round(p, 4)})

        if not chunk_probs:
            raise ValueError("no usable chunks (no faces detected or no audio segments)")

        probs = np.asarray(chunk_probs, dtype=np.float64)
        session_prob = float(probs.mean())
        pred = 1 if session_prob >= float(threshold) else 0

        result["success"] = True
        result["probability"] = session_prob
        result["prediction"] = int(pred)
        result["prediction_label"] = "experimental" if pred == 1 else "control"
        result["risk_level"] = "high" if session_prob >= 0.5 else "medium" if session_prob >= 0.4 else "low"
        result["feature_source"] = "vgg16" if used_vgg else "fallback_sobel"
        result["chunks_used"] = len(chunk_probs)
        result["positive_chunk_ratio"] = float((probs >= 0.5).mean())
        result["chunk_timeline"] = timeline
        return result

    except Exception as e:
        result["error"] = str(e)
        return result

