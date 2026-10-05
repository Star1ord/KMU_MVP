#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze a newly uploaded media file and return multimodal predictions.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.preprocessing.process_new_media import process_new_media
from src.utils.model_loading import ModelLoadError, first_existing_path, safe_load_serialized_model
from src.utils.paths import AUDIO_WAV_DIR, NLP_MODELS_DIR


def _wrap_result(result: Dict[str, Any]) -> Dict[str, Any]:
    pretty = build_pretty_response(result)
    return {"formatted_result": pretty, "raw_result": result}


def _risk_level(score: float) -> str:
    if score >= 0.5:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _append_error(result: Dict[str, Any], message: str) -> None:
    existing = result.get("error")
    if existing:
        result["error"] = f"{existing} | {message}"
    else:
        result["error"] = message


def _candidate_nlp_model_paths() -> list[Path]:
    return [
        NLP_MODELS_DIR / "late_fusion_v3_clean.pkl",
        ROOT / "models" / "nlp" / "late_fusion_v3_clean.pkl",
        ROOT / "models" / "late_fusion_v3_clean.pkl",
    ]


def _extract_session_rows(df: pd.DataFrame, session_id: str, label: int) -> tuple[str | None, pd.DataFrame]:
    available_file_ids = df["file_id"].astype(str).unique().tolist() if "file_id" in df.columns else []
    search_variants = [
        session_id,
        f"{label}_{session_id}",
        session_id.replace("0_", "").replace("1_", ""),
        f"0_{session_id}",
        f"1_{session_id}",
    ]

    for variant in search_variants:
        session_rows = df[df["file_id"].astype(str) == str(variant)].copy()
        if not session_rows.empty:
            return str(variant), session_rows

    for file_id in available_file_ids:
        file_id_str = str(file_id)
        if session_id in file_id_str or file_id_str in session_id:
            session_rows = df[df["file_id"].astype(str) == file_id_str].copy()
            if not session_rows.empty:
                return file_id_str, session_rows

    return None, pd.DataFrame()


def _aggregate_audio_features(model_package: Dict[str, Any], session_data: pd.DataFrame) -> Dict[str, float]:
    audio_cols = model_package["audio_feature_names"]
    base_features: dict[str, list[tuple[str, str]]] = {}

    for col in audio_cols:
        match = re.match(r"audio_(.+)_(mean|std|median|min|max)$", col)
        if not match:
            continue

        base_name = match.group(1)
        stat = match.group(2)
        base_features.setdefault(base_name, []).append((stat, col))

    aggregated: dict[str, float] = {}
    for base_name, stat_columns in base_features.items():
        values = (
            session_data[base_name].values
            if base_name in session_data.columns
            else np.array([], dtype=float)
        )
        values = values[~pd.isna(values)]

        for stat, column_name in stat_columns:
            if len(values) == 0:
                aggregated[column_name] = 0.0
            elif stat == "mean":
                aggregated[column_name] = float(np.mean(values))
            elif stat == "std":
                aggregated[column_name] = float(np.std(values)) if len(values) > 1 else 0.0
            elif stat == "median":
                aggregated[column_name] = float(np.median(values))
            elif stat == "min":
                aggregated[column_name] = float(np.min(values))
            elif stat == "max":
                aggregated[column_name] = float(np.max(values))

    for column_name in audio_cols:
        aggregated.setdefault(column_name, 0.0)

    return aggregated


def _predict_positive_probability(model: Any, features: Any) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        return float(probabilities.reshape(probabilities.shape[0], -1)[0, -1])

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features), dtype=float).reshape(-1)[0]
        return float(1.0 / (1.0 + np.exp(-decision)))

    prediction = np.asarray(model.predict(features), dtype=float).reshape(-1)[0]
    return float(prediction)


def _predict_positive_probabilities(model: Any, features: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        return probabilities.reshape(probabilities.shape[0], -1)[:, -1]

    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features), dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-decision))

    prediction = np.asarray(model.predict(features), dtype=float).reshape(-1)
    return prediction


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(numeric) or np.isinf(numeric):
        return 0.0
    return float(max(0.0, min(1.0, numeric)))


def _normalize_series(values: pd.Series, invert: bool = False) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    valid = numeric.dropna()
    if valid.empty:
        normalized = pd.Series(np.zeros(len(numeric), dtype=float), index=numeric.index)
    else:
        lower = float(valid.min())
        upper = float(valid.max())
        if np.isclose(lower, upper):
            normalized = pd.Series(np.full(len(numeric), 0.5, dtype=float), index=numeric.index)
        else:
            normalized = (numeric - lower) / (upper - lower)
        normalized = normalized.fillna(0.0).clip(lower=0.0, upper=1.0)

    if invert:
        normalized = 1.0 - normalized

    return normalized.astype(float)


def _build_text_segments(
    rows: pd.DataFrame,
    vectorizer: Any,
    classifier: Any,
) -> list[Dict[str, Any]]:
    if rows.empty or "text" not in rows.columns:
        return []

    prepared_rows = rows.copy()
    prepared_rows["text"] = prepared_rows["text"].fillna("").astype(str)
    prepared_rows = prepared_rows[prepared_rows["text"].str.strip() != ""].reset_index(drop=True)
    if prepared_rows.empty:
        return []

    x_tfidf = vectorizer.transform(prepared_rows["text"].tolist())
    scores = _predict_positive_probabilities(classifier, x_tfidf)

    segments: list[Dict[str, Any]] = []
    for index, row in prepared_rows.iterrows():
        start_time = _coerce_time_value(row.get("start", row.get("start_time")), 0.0)
        end_time = _coerce_time_value(row.get("end", row.get("end_time")), start_time)
        segment_score = float(scores[index]) if index < len(scores) else 0.0
        segments.append(
            {
                "segment_id": str(row.get("segment_id", f"text_segment_{index:04d}")),
                "start_time": start_time,
                "end_time": end_time,
                "duration": max(0.0, end_time - start_time),
                "text": str(row.get("text", "")).strip(),
                "risk_score": segment_score,
                "risk_level": _risk_level(segment_score),
                "top_features": [],
                "all_features": {},
            }
        )

    return segments


def _build_video_interpretation(video_result: Dict[str, Any] | None) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    if not video_result:
        return [], []

    csv_path = video_result.get("csv_path")
    if not csv_path or not Path(str(csv_path)).exists():
        return [], []

    try:
        frame_df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return [], []

    if frame_df.empty:
        return [], []

    source_map = {
        "gaze_away_time": ("Gaze away", False),
        "head_down_duration": ("Head down", False),
        "face_touch_count": ("Face touch", False),
        "blink_count": ("Blink rate", False),
        "motion": ("Motion", False),
        "head_stability": ("Head stability", True),
    }

    normalized_columns: dict[str, pd.Series] = {}
    for column_name, (_label, invert) in source_map.items():
        if column_name in frame_df.columns:
            normalized_columns[column_name] = _normalize_series(frame_df[column_name], invert=invert)

    if not normalized_columns:
        return [], []

    score_components = {
        "gaze_away_time": 0.28,
        "head_down_duration": 0.22,
        "face_touch_count": 0.18,
        "blink_count": 0.12,
        "motion": 0.10,
        "head_stability": 0.10,
    }

    frame_scores = pd.Series(np.zeros(len(frame_df), dtype=float), index=frame_df.index)
    for column_name, weight in score_components.items():
        if column_name in normalized_columns:
            frame_scores = frame_scores + (normalized_columns[column_name] * weight)

    timestamp_values = (
        pd.to_numeric(frame_df["timestamp"], errors="coerce")
        if "timestamp" in frame_df.columns
        else pd.to_numeric(frame_df.get("frame_idx"), errors="coerce")
    ).fillna(0.0)

    timeline: list[Dict[str, Any]] = []
    for index, score in frame_scores.items():
        timeline.append(
            {
                "x": float(timestamp_values.loc[index]),
                "y": _clamp01(score),
            }
        )

    top_rows = frame_scores.sort_values(ascending=False).head(4).index.tolist()
    markers: list[Dict[str, Any]] = []
    for row_index in top_rows:
        driver_pairs: list[tuple[str, float]] = []
        for column_name, (label, _invert) in source_map.items():
            series = normalized_columns.get(column_name)
            if series is None:
                continue
            driver_pairs.append((label, _clamp01(series.loc[row_index])))
        driver_pairs.sort(key=lambda item: item[1], reverse=True)
        top_signals = [
            {"feature": label, "value": float(score)}
            for label, score in driver_pairs[:3]
            if score > 0.0
        ]
        markers.append(
            {
                "timestamp": float(timestamp_values.loc[row_index]),
                "frame_idx": int(frame_df.loc[row_index].get("frame_idx", row_index)),
                "score": _clamp01(frame_scores.loc[row_index]),
                "top_signals": top_signals,
            }
        )

    return timeline, markers


def _build_acoustic_interpretation(session_data: pd.DataFrame | None) -> tuple[Dict[str, Any] | None, list[Dict[str, Any]], list[str]]:
    if session_data is None or session_data.empty:
        return None, [], []

    required_any = {"pause_ratio", "speech_rate", "articulation_rate", "asr_conf_mean"}
    if not any(column in session_data.columns for column in required_any):
        return None, [], []

    numeric = session_data.copy()

    pause_series = _normalize_series(numeric["pause_ratio"]) if "pause_ratio" in numeric.columns else pd.Series(np.zeros(len(numeric)), index=numeric.index)
    slow_speech_series = _normalize_series(numeric["speech_rate"], invert=True) if "speech_rate" in numeric.columns else pd.Series(np.zeros(len(numeric)), index=numeric.index)
    low_articulation_series = _normalize_series(numeric["articulation_rate"], invert=True) if "articulation_rate" in numeric.columns else pd.Series(np.zeros(len(numeric)), index=numeric.index)
    low_confidence_series = _normalize_series(numeric["asr_conf_mean"], invert=True) if "asr_conf_mean" in numeric.columns else pd.Series(np.zeros(len(numeric)), index=numeric.index)
    energy_variation_series = _normalize_series(numeric["rms_mean"]) if "rms_mean" in numeric.columns else pd.Series(np.zeros(len(numeric)), index=numeric.index)

    segment_scores = (
        pause_series * 0.35
        + slow_speech_series * 0.25
        + low_articulation_series * 0.20
        + low_confidence_series * 0.10
        + energy_variation_series * 0.10
    ).clip(lower=0.0, upper=1.0)

    timeline: list[Dict[str, Any]] = []
    for index, row in numeric.iterrows():
        start_time = _coerce_time_value(row.get("start", row.get("start_time")), 0.0)
        end_time = _coerce_time_value(row.get("end", row.get("end_time")), start_time)
        midpoint = start_time + max(0.0, end_time - start_time) / 2.0
        timeline.append({"x": midpoint, "y": _clamp01(segment_scores.loc[index])})

    profile = {
        "labels": [
            "Long pauses",
            "Slow speech",
            "Low articulation",
            "Low ASR confidence",
            "Energy variation",
            "Acoustic risk index",
        ],
        "values": [
            float(pause_series.mean()),
            float(slow_speech_series.mean()),
            float(low_articulation_series.mean()),
            float(low_confidence_series.mean()),
            float(energy_variation_series.mean()),
            float(segment_scores.mean()),
        ],
    }

    insights: list[str] = []
    if profile["values"][0] >= 0.55:
        insights.append("Long pauses are frequent in this interview.")
    if profile["values"][1] >= 0.55:
        insights.append("Speech pace is slower than the rest of this session.")
    if profile["values"][2] >= 0.55:
        insights.append("Articulation becomes less fluent in several segments.")
    if profile["values"][3] >= 0.45:
        insights.append("Speech confidence drops in multiple transcript segments.")
    if profile["values"][5] < 0.35:
        insights.append("Acoustic pattern looks relatively stable across the session.")

    return profile, timeline, insights


def _apply_primary_result(result: Dict[str, Any], available_results: list[Dict[str, Any]]) -> None:
    ensemble = result.get("ensemble_result")
    if ensemble and len(available_results) >= 2:
        result["prediction"] = int(ensemble["ensemble_prediction"])
        result["risk_score"] = float(ensemble["ensemble_score"])
        result["risk_level"] = ensemble["risk_level"]
        result["prediction_label"] = ensemble["ensemble_prediction_label"]
        result["overall_source"] = "ensemble"
        result["success"] = True
        return

    _apply_multimodal_fallback(result, available_results)


def _coerce_time_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_transcript_text(transcript_path: str | Path | None) -> tuple[str, list[Dict[str, Any]]]:
    if not transcript_path:
        return "", []

    path = Path(transcript_path)
    if not path.exists():
        return "", []

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        return "", []

    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        return "", []

    texts: list[str] = []
    normalized_segments: list[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue

        text = str(segment.get("text", "")).strip()
        if text:
            texts.append(text)

        start_time = _coerce_time_value(segment.get("start"), 0.0)
        end_time = _coerce_time_value(segment.get("end"), start_time)
        normalized_segments.append(
            {
                "segment_id": str(segment.get("id", f"transcript_segment_{index:04d}")),
                "start_time": start_time,
                "end_time": end_time,
                "duration": max(0.0, end_time - start_time),
                "text": text,
                "risk_score": None,
                "risk_level": None,
                "top_features": [],
                "all_features": {},
            }
        )

    return " ".join(texts).strip(), normalized_segments


def _resolve_transcript_path(processing_report: Dict[str, Any] | None, session_id: str) -> Path | None:
    if not processing_report:
        return None

    transcript_paths = processing_report.get("transcript_paths", {})
    if not isinstance(transcript_paths, dict):
        return None

    candidate = transcript_paths.get(session_id)
    if candidate and Path(candidate).exists():
        return Path(candidate)

    for value in transcript_paths.values():
        path = Path(value)
        if path.exists() and session_id in path.stem:
            return path

    return None


def _run_nlp_prediction(
    result: Dict[str, Any],
    session_id: str,
    session_data: pd.DataFrame,
) -> None:
    model_path = first_existing_path(_candidate_nlp_model_paths())
    if model_path is None:
        raise FileNotFoundError(
            f"NLP model not found. Checked: {[str(path) for path in _candidate_nlp_model_paths()]}"
        )

    model_package = safe_load_serialized_model(model_path, model_name="NLP late_fusion_v3_clean")
    print(f"NLP model loaded: {model_path}")

    audio_cols = model_package["audio_feature_names"]
    audio_features = _aggregate_audio_features(model_package, session_data)
    full_text = " ".join(session_data["text"].fillna("").astype(str).tolist()).strip()

    text_model = model_package["text_model"]
    audio_model = model_package["audio_model"]
    meta_model = model_package["meta_model"]
    threshold = float(model_package["threshold"])

    x_tfidf = text_model["vectorizer"].transform([full_text])
    p_text = _predict_positive_probability(text_model["classifier"], x_tfidf)
    text_segments = _build_text_segments(session_data, text_model["vectorizer"], text_model["classifier"])

    x_audio = np.array([[audio_features[col] for col in audio_cols]], dtype=float)
    x_audio = np.nan_to_num(x_audio, nan=0.0, posinf=0.0, neginf=0.0)
    p_audio = _predict_positive_probability(audio_model, x_audio)

    x_meta = np.array([[p_text, p_audio]], dtype=float)
    p_final = _predict_positive_probability(meta_model, x_meta)
    final_prediction = int(p_final >= threshold)

    print("\n[Late Fusion V3 Clean Prediction]")
    print(f"  Text score: {p_text:.3f}")
    print(f"  Audio score: {p_audio:.3f}")
    print(f"  Final score: {p_final:.3f}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Prediction: {'RISK' if final_prediction == 1 else 'CONTROL'}")

    result["nlp_success"] = True
    result["success"] = True
    result["nlp_prediction"] = final_prediction
    result["nlp_risk_score"] = p_final
    result["nlp_risk_level"] = "high" if p_final >= 0.5 else "medium" if p_final >= 0.4 else "low"
    result["nlp_prediction_label"] = "experimental" if final_prediction == 1 else "control"
    result["prediction"] = final_prediction
    result["risk_score"] = p_final
    result["risk_level"] = result["nlp_risk_level"]
    result["prediction_label"] = result["nlp_prediction_label"]
    result["full_text"] = full_text
    result["transcript"] = full_text
    result["overall_source"] = "nlp"
    result["text_segments"] = text_segments


def _run_text_only_nlp_prediction(
    result: Dict[str, Any],
    transcript_path: str | Path,
) -> None:
    model_path = first_existing_path(_candidate_nlp_model_paths())
    if model_path is None:
        raise FileNotFoundError(
            f"NLP model not found. Checked: {[str(path) for path in _candidate_nlp_model_paths()]}"
        )

    full_text, transcript_segments = _load_transcript_text(transcript_path)
    if not full_text:
        raise FileNotFoundError(f"Transcript text not found in {transcript_path}")

    model_package = safe_load_serialized_model(model_path, model_name="NLP late_fusion_v3_clean")
    print(f"NLP model loaded: {model_path}")

    text_model = model_package["text_model"]
    x_tfidf = text_model["vectorizer"].transform([full_text])
    p_text = _predict_positive_probability(text_model["classifier"], x_tfidf)
    if transcript_segments:
        transcript_rows = pd.DataFrame(
            [
                {
                    "segment_id": segment.get("segment_id"),
                    "start": segment.get("start_time"),
                    "end": segment.get("end_time"),
                    "text": segment.get("text", ""),
                }
                for segment in transcript_segments
            ]
        )
        transcript_segments = _build_text_segments(
            transcript_rows,
            text_model["vectorizer"],
            text_model["classifier"],
        )
    threshold = 0.5
    prediction = int(p_text >= threshold)

    print("\n[Transcript-only NLP Fallback]")
    print(f"  Text score: {p_text:.3f}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Prediction: {'RISK' if prediction == 1 else 'CONTROL'}")

    result["nlp_success"] = True
    result["success"] = True
    result["nlp_prediction"] = prediction
    result["nlp_risk_score"] = p_text
    result["nlp_risk_level"] = "high" if p_text >= 0.5 else "medium" if p_text >= 0.4 else "low"
    result["nlp_prediction_label"] = "experimental" if prediction == 1 else "control"
    result["prediction"] = prediction
    result["risk_score"] = p_text
    result["risk_level"] = result["nlp_risk_level"]
    result["prediction_label"] = result["nlp_prediction_label"]
    result["full_text"] = full_text
    result["transcript"] = full_text
    result["overall_source"] = "nlp_text_fallback"
    result["text_segments"] = transcript_segments

    if transcript_segments and not result.get("segments"):
        result["segments"] = transcript_segments


def _try_text_only_nlp_fallback(
    result: Dict[str, Any],
    transcript_path: str | Path | None,
    reason: str,
) -> bool:
    if transcript_path is None or not Path(transcript_path).exists():
        return False

    try:
        _run_text_only_nlp_prediction(result, transcript_path)
        result["nlp_error"] = reason
        return True
    except Exception as exc:
        result["nlp_error"] = f"{reason} Transcript-only fallback failed: {exc}"
        print(f"NLP fallback activated: {exc}")
        return False


def _run_segment_analysis(result: Dict[str, Any], session_id: str, session_data: pd.DataFrame) -> None:
    try:
        from src.inference.segment_analyzer import SegmentAnalyzer

        analyzer = SegmentAnalyzer()
        missing_features = [feature for feature in analyzer.feature_names if feature not in session_data.columns]
        if missing_features:
            missing_ratio = len(missing_features) / max(1, len(analyzer.feature_names))
            if missing_ratio > 0.25:
                print(
                    "Warning: Skipping segment analysis because "
                    f"{len(missing_features)}/{len(analyzer.feature_names)} required features are unavailable."
                )
                result["segments"] = []
                return

        analysis = analyzer.analyze_session(session_data, session_id)

        feature_names = analyzer.feature_names
        audio_features = analyzer.audio_features
        text_features = analyzer.text_features

        for index, segment in enumerate(analysis.segments):
            seg_row = session_data.iloc[index] if index < len(session_data) else None
            all_features: dict[str, Dict[str, Any]] = {}

            if seg_row is not None:
                for feature_name in feature_names:
                    if feature_name not in session_data.columns:
                        continue

                    feature_value = seg_row[feature_name]
                    all_features[feature_name] = {
                        "value": float(feature_value) if pd.notna(feature_value) else 0.0,
                        "type": "audio" if feature_name in audio_features else "text",
                    }

            result["segments"].append(
                {
                    "segment_id": segment.segment_id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "duration": segment.duration,
                    "text": segment.text,
                    "risk_score": segment.risk_score,
                    "risk_level": segment.risk_level,
                    "top_features": segment.top_features[:5],
                    "all_features": all_features,
                }
            )

        result["feature_names"] = feature_names
        result["audio_features"] = audio_features
        result["text_features"] = text_features
    except Exception as exc:  # pragma: no cover - best effort for interpretability
        print(f"Warning: Segment analysis failed: {exc}")
        result["segments"] = []


def _build_available_results(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    available_results: list[Dict[str, Any]] = []

    if result.get("nlp_success") and result.get("risk_score") is not None:
        available_results.append(
            {
                "name": "nlp",
                "score": float(result["risk_score"]),
                "prediction": int(result["prediction"]),
            }
        )

    video_result = result.get("video_result")
    if video_result and video_result.get("success"):
        available_results.append(
            {
                "name": "video",
                "score": float(video_result["probability"]),
                "prediction": int(video_result["prediction"]),
            }
        )

    cv_audio_result = result.get("cv_audio_result")
    if cv_audio_result and cv_audio_result.get("success"):
        available_results.append(
            {
                "name": "cv_audio",
                "score": float(cv_audio_result["probability"]),
                "prediction": int(cv_audio_result["prediction"]),
            }
        )

    return available_results


def _build_ensemble_result(available_results: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not available_results:
        return None

    if len(available_results) == 1:
        item = available_results[0]
        return {
            "ensemble_score": float(item["score"]),
            "ensemble_prediction": int(item["prediction"]),
            "ensemble_prediction_label": "experimental" if int(item["prediction"]) == 1 else "control",
            "models_used": [item["name"]],
            "individual_scores": {item["name"]: float(item["score"])},
            "individual_predictions": {item["name"]: int(item["prediction"])},
            "agreement": True,
            "risk_level": _risk_level(float(item["score"])),
        }

    scores = [float(item["score"]) for item in available_results]
    predictions = [int(item["prediction"]) for item in available_results]
    ensemble_score = float(np.mean(scores))
    ensemble_prediction = int(ensemble_score >= 0.5)

    print("\n[Ensemble Prediction]")
    for item in available_results:
        print(f"  {item['name']} score: {float(item['score']):.3f}")
    print(f"  Ensemble score: {ensemble_score:.3f}")
    print(f"  Ensemble prediction: {'RISK' if ensemble_prediction == 1 else 'CONTROL'}")

    return {
        "ensemble_score": ensemble_score,
        "ensemble_prediction": ensemble_prediction,
        "ensemble_prediction_label": "experimental" if ensemble_prediction == 1 else "control",
        "models_used": [item["name"] for item in available_results],
        "individual_scores": {item["name"]: float(item["score"]) for item in available_results},
        "individual_predictions": {item["name"]: int(item["prediction"]) for item in available_results},
        "agreement": len(set(predictions)) == 1,
        "risk_level": _risk_level(ensemble_score),
    }


def _resolve_audio_path(
    original_media_path: str,
    processing_report: Dict[str, Any] | None,
    session_data: pd.DataFrame | None,
    session_id: str,
    label: int,
) -> str | None:
    original_path = Path(original_media_path)
    if original_path.exists() and original_path.suffix.lower() in {".wav", ".mp3", ".m4a"}:
        return str(original_path)

    if processing_report:
        wav_paths = processing_report.get("wav_paths", {})
        if session_id in wav_paths and Path(wav_paths[session_id]).exists():
            return str(Path(wav_paths[session_id]))

    expected_wav = AUDIO_WAV_DIR / str(label) / f"{session_id}.wav"
    if expected_wav.exists():
        return str(expected_wav)

    if session_data is not None and "audio_path" in session_data.columns:
        audio_paths = session_data["audio_path"].dropna().unique()
        for audio_path in audio_paths:
            candidate = Path(str(audio_path))
            if candidate.exists():
                return str(candidate)

    return None


def _run_video_models(
    media_file: Path,
    skip_video: bool,
    skip_cv_audio: bool,
    video_sample_every: int,
    cv_audio_sample_rate: float,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    video_result = None
    cv_audio_result = None

    is_video = media_file.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    if not is_video:
        return video_result, cv_audio_result

    if not skip_video:
        try:
            from inference.video_ensemble import process_video_for_prediction, predict_with_video_model

            print("\n" + "=" * 80)
            print("VIDEO PROCESSING (independent from audio+NLP pipeline)")
            print("=" * 80)

            video_csv_path, video_error = process_video_for_prediction(
                video_path=str(media_file),
                output_dir=None,
                sample_every=max(1, int(video_sample_every)),
                use_emotions=None,
            )

            if video_csv_path and not video_error:
                print(f"Video features CSV created: {video_csv_path}")
                candidate_result = predict_with_video_model(
                    video_csv_path=video_csv_path,
                    model_path=None,
                    threshold=0.6,
                )
                candidate_result.setdefault("csv_path", video_csv_path)
                if candidate_result.get("success"):
                    video_result = candidate_result
                    print("\n[Video Model Prediction]")
                    print(f"  Video score: {video_result['probability']:.3f}")
                    print(f"  Prediction: {'RISK' if video_result['prediction'] == 1 else 'CONTROL'}")
                    print(f"  Risk level: {video_result['risk_level']}")
                else:
                    print(f"Video prediction failed: {candidate_result.get('error', 'Unknown error')}")
            else:
                print(f"Video processing failed: {video_error}")
        except Exception as exc:  # pragma: no cover - optional path
            print(f"Error in video processing: {exc}")

    if not skip_cv_audio:
        try:
            from src.pipeline.cv_audio_pipeline import predict_cv_audio as predict_cv_audio_func

            print("\n" + "=" * 80)
            print("CV+AUDIO PROCESSING")
            print("=" * 80)

            candidate_result = predict_cv_audio_func(
                video_path=str(media_file),
                model_path=None,
                threshold=0.5,
                sample_rate=float(cv_audio_sample_rate),
            )
            if candidate_result.get("success"):
                cv_audio_result = candidate_result
                print("\n[CV+Audio Model Prediction]")
                print(f"  CV+Audio score: {cv_audio_result['probability']:.3f}")
                print(f"  Prediction: {'RISK' if cv_audio_result['prediction'] == 1 else 'CONTROL'}")
                print(f"  Risk level: {cv_audio_result['risk_level']}")
            else:
                print(f"CV+Audio prediction failed: {candidate_result.get('error', 'Unknown error')}")
        except Exception as exc:  # pragma: no cover - optional path
            print(f"Error in CV+Audio processing: {exc}")

    return video_result, cv_audio_result


def _apply_multimodal_fallback(result: Dict[str, Any], available_results: list[Dict[str, Any]]) -> None:
    if result.get("nlp_success") or not available_results:
        return

    primary = result.get("ensemble_result")
    if primary:
        result["prediction"] = int(primary["ensemble_prediction"])
        result["risk_score"] = float(primary["ensemble_score"])
        result["risk_level"] = primary["risk_level"]
        result["prediction_label"] = primary["ensemble_prediction_label"]
        result["overall_source"] = "ensemble"
    else:
        item = available_results[0]
        result["prediction"] = int(item["prediction"])
        result["risk_score"] = float(item["score"])
        result["risk_level"] = _risk_level(float(item["score"]))
        result["prediction_label"] = "experimental" if int(item["prediction"]) == 1 else "control"
        result["overall_source"] = item["name"]

    result["success"] = True


def analyze_new_media_file(
    media_path: str,
    label: Optional[int] = None,
    add_to_training: bool = False,
    whisper_model: str = "medium",
    skip_transcription: bool = False,
    skip_video: bool = False,
    skip_cv_audio: bool = False,
    video_sample_every: int = 8,
    cv_audio_sample_rate: float = 0.25,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "success": False,
        "nlp_success": False,
        "session_id": None,
        "nlp_prediction": None,
        "nlp_risk_score": None,
        "nlp_risk_level": None,
        "nlp_prediction_label": None,
        "prediction": None,
        "risk_score": None,
        "risk_level": None,
        "prediction_label": None,
        "full_text": "",
        "transcript": "",
        "segments": [],
        "text_segments": [],
        "features_csv": None,
        "audio_path": None,
        "error": None,
        "video_result": None,
        "cv_audio_result": None,
        "video_timeline": [],
        "video_markers": [],
        "acoustic_profile": None,
        "acoustic_timeline": [],
        "acoustic_insights": [],
        "ensemble_result": None,
        "processing_report": None,
        "nlp_error": None,
        "overall_source": None,
    }

    media_file = Path(media_path).resolve()
    if not media_file.exists():
        result["error"] = f"File not found: {media_file}"
        return _wrap_result(result)

    if label is None:
        label = 0

    session_id = media_file.stem
    result["session_id"] = session_id
    original_media_path = str(media_file)

    temp_dir: Path | None = None

    try:
        print("\n" + "=" * 80)
        print(f"Analyzing: {session_id}")
        print("=" * 80)

        processing_report = process_new_media(
            media_files=[str(media_file)],
            label=label,
            whisper_model=whisper_model,
            whisper_device="cpu",
            skip_pipeline=False,
            add_to_training=add_to_training,
            skip_transcription=skip_transcription,
            return_details=True,
        )
        result["processing_report"] = processing_report

        pipeline_errors = processing_report.get("errors", [])
        if pipeline_errors:
            print("warning: pipeline reported issues for this session")
            for message in pipeline_errors:
                print(f"  - {message}")
            _append_error(result, " | ".join(str(message) for message in pipeline_errors))

        video_result, cv_audio_result = _run_video_models(
            media_file=media_file,
            skip_video=skip_video,
            skip_cv_audio=skip_cv_audio,
            video_sample_every=video_sample_every,
            cv_audio_sample_rate=cv_audio_sample_rate,
        )
        result["video_result"] = video_result
        result["cv_audio_result"] = cv_audio_result
        result["video_timeline"], result["video_markers"] = _build_video_interpretation(video_result)

        merged_features_path = processing_report.get("merged_features_path")
        session_data: pd.DataFrame | None = None
        transcript_path = _resolve_transcript_path(processing_report, session_id)

        segments_found_for = set(processing_report.get("segments_found_for", []))
        can_use_nlp_features = not processing_report.get("errors") or session_id in segments_found_for

        if merged_features_path and Path(merged_features_path).exists() and can_use_nlp_features:
            df = pd.read_csv(merged_features_path, low_memory=False)
            print(f"Loaded {len(df)} total segments from merged_features.csv")

            found_session_id, session_rows = _extract_session_rows(df, session_id, label)
            if session_rows.empty:
                if not _try_text_only_nlp_fallback(
                    result,
                    transcript_path,
                    "Late-fusion segment features were not found for this session; used transcript-only NLP fallback.",
                ):
                    result["nlp_error"] = (
                        f"No merged segment rows found for session '{session_id}'. "
                        f"Check {processing_report.get('segments_metadata_path')} and WhisperX logs."
                    )
                    _append_error(result, result["nlp_error"])
            else:
                session_data = session_rows
                if found_session_id:
                    session_id = found_session_id
                    result["session_id"] = session_id
                print(f"Found session data with file_id: {session_id} ({len(session_data)} segments)")
                result["features_csv"] = merged_features_path

                try:
                    _run_nlp_prediction(result, session_id, session_data)
                    _run_segment_analysis(result, session_id, session_data)
                    if not result.get("segments") and result.get("text_segments"):
                        result["segments"] = result["text_segments"]
                except (FileNotFoundError, ModelLoadError) as exc:
                    print(f"NLP fallback activated: {exc}")
                    if not _try_text_only_nlp_fallback(
                        result,
                        transcript_path,
                        f"Late-fusion NLP failed ({exc}); used transcript-only NLP fallback.",
                    ):
                        result["nlp_error"] = str(exc)
                        _append_error(result, f"NLP fallback activated: {exc}")
                except Exception as exc:
                    print(f"NLP fallback activated: {exc}")
                    if not _try_text_only_nlp_fallback(
                        result,
                        transcript_path,
                        f"NLP late-fusion analysis failed ({exc}); used transcript-only NLP fallback.",
                    ):
                        result["nlp_error"] = f"NLP analysis failed: {exc}"
                        _append_error(result, result["nlp_error"])
        else:
            if not _try_text_only_nlp_fallback(
                result,
                transcript_path,
                "Session-specific audio fusion features are unavailable; used transcript-only NLP fallback.",
            ):
                result["nlp_error"] = (
                    "Session-specific NLP features are unavailable because the current pipeline run "
                    "did not produce valid segments_metadata.csv rows for this session."
                )
                _append_error(result, result["nlp_error"])

        result["acoustic_profile"], result["acoustic_timeline"], result["acoustic_insights"] = _build_acoustic_interpretation(session_data)
        available_results = _build_available_results(result)
        result["ensemble_result"] = _build_ensemble_result(available_results)
        _apply_primary_result(result, available_results)

        result["audio_path"] = _resolve_audio_path(
            original_media_path=original_media_path,
            processing_report=processing_report,
            session_data=session_data,
            session_id=session_id,
            label=label,
        )

        if result.get("success"):
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print("=" * 80)
            print(f"Session: {result['session_id']}")
            print(f"Prediction: {'RISK' if int(result['prediction']) == 1 else 'CONTROL'}")
            print(f"Risk Score: {float(result['risk_score']) * 100:.1f}%")
            print(f"Segments: {len(result['segments'])}")
            print(f"High-risk segments: {sum(1 for segment in result['segments'] if segment['risk_level'] == 'high')}")
        elif not result.get("error"):
            result["error"] = "No model produced a usable result for this session"
    finally:
        if temp_dir is not None:
            try:
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temporary files: {temp_dir}")
            except Exception as exc:  # pragma: no cover - best effort cleanup
                print(f"Warning: Could not clean up temp dir: {exc}")

    return _wrap_result(result)


def build_pretty_response(result: Dict[str, Any]) -> Dict[str, Any]:
    ensemble = result.get("ensemble_result")
    nlp_prediction = result.get("nlp_prediction") if result.get("nlp_success") else None
    nlp_prediction_label = result.get("nlp_prediction_label") if result.get("nlp_success") else None

    models_summary = {
        "nlp": {
            "success": bool(result.get("nlp_success", False)),
            "prediction": int(nlp_prediction) if nlp_prediction is not None else None,
            "prediction_label": nlp_prediction_label,
            "risk_level": result.get("nlp_risk_level") if result.get("nlp_success") else None,
            "probability": float(result.get("nlp_risk_score")) if result.get("nlp_success") and result.get("nlp_risk_score") is not None else None,
            "transcript": result.get("transcript", ""),
            "full_text": result.get("full_text", ""),
            "error": result.get("nlp_error"),
        }
    }

    cv_audio_result = result.get("cv_audio_result")
    models_summary["cv_audio"] = (
        {
            "success": bool(cv_audio_result.get("success", False)),
            "prediction": int(cv_audio_result.get("prediction")) if cv_audio_result.get("prediction") is not None else None,
            "prediction_label": cv_audio_result.get("prediction_label"),
            "risk_level": cv_audio_result.get("risk_level"),
            "probability": float(cv_audio_result.get("probability")) if cv_audio_result.get("probability") is not None else None,
            "metrics": cv_audio_result.get("metrics"),
            "error": cv_audio_result.get("error"),
        }
        if cv_audio_result
        else None
    )

    video_result = result.get("video_result")
    models_summary["cv"] = (
        {
            "success": bool(video_result.get("success", False)),
            "prediction": int(video_result.get("prediction")) if video_result.get("prediction") is not None else None,
            "prediction_label": video_result.get("prediction_label"),
            "risk_level": video_result.get("risk_level"),
            "probability": float(video_result.get("probability")) if video_result.get("probability") is not None else None,
            "metrics": video_result.get("metrics"),
            "error": video_result.get("error"),
        }
        if video_result
        else None
    )

    overall = {
        "success": bool(result.get("success", False)),
        "prediction": int(result.get("prediction")) if result.get("prediction") is not None else None,
        "prediction_label": result.get("prediction_label"),
        "risk_level": result.get("risk_level"),
        "risk_score": float(result.get("risk_score")) if result.get("risk_score") is not None else None,
        "source": result.get("overall_source"),
        "ensemble": None,
    }

    if ensemble:
        overall["ensemble"] = {
            "ensemble_score": float(ensemble.get("ensemble_score")),
            "ensemble_prediction": int(ensemble.get("ensemble_prediction")),
            "ensemble_prediction_label": ensemble.get("ensemble_prediction_label"),
            "models_used": ensemble.get("models_used"),
            "individual_scores": ensemble.get("individual_scores"),
            "individual_predictions": ensemble.get("individual_predictions"),
            "agreement": bool(ensemble.get("agreement")),
            "risk_level": ensemble.get("risk_level"),
        }

    return {
        "success": bool(result.get("success", False)),
        "session_id": result.get("session_id"),
        "overall": overall,
        "models": models_summary,
        "transcript": result.get("transcript", ""),
        "full_text": result.get("full_text", ""),
        "raw_text": result.get("full_text", ""),
        "segments": result.get("segments", []),
        "features_csv": result.get("features_csv"),
        "audio_path": result.get("audio_path"),
        "video_timeline": result.get("video_timeline", []),
        "video_markers": result.get("video_markers", []),
        "acoustic_profile": result.get("acoustic_profile"),
        "acoustic_timeline": result.get("acoustic_timeline", []),
        "acoustic_insights": result.get("acoustic_insights", []),
        "processing_report": result.get("processing_report"),
        "error": result.get("error"),
    }


def analyze_multiple_media_files(
    media_paths: List[str],
    label: Optional[int] = None,
    add_to_training: bool = False,
    whisper_model: str = "medium",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for media_path in media_paths:
        print("\n" + "#" * 80)
        print(f"Processing: {Path(media_path).name}")
        print("#" * 80)

        wrapped = analyze_new_media_file(
            media_path,
            label=label,
            add_to_training=add_to_training,
            whisper_model=whisper_model,
        )
        results.append(wrapped["raw_result"])

        if not wrapped["raw_result"].get("success"):
            print(f"ERROR: {wrapped['raw_result'].get('error')}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files processed: {len(results)}")
    print(f"Successful: {sum(1 for item in results if item.get('success'))}")
    print(f"Failed: {sum(1 for item in results if not item.get('success'))}")

    return results


def export_analysis_report(result: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Report saved: {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze new media files with segment-level predictions")
    parser.add_argument("media_files", nargs="+", help="Media files to analyze")
    parser.add_argument("--label", type=int, choices=[0, 1], help="Label (0=control, 1=risk)")
    parser.add_argument("--add-to-training", action="store_true", help="Add to training dataset")
    parser.add_argument("--whisper-model", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--export", help="Export report to JSON file")

    args = parser.parse_args()

    if len(args.media_files) == 1:
        wrapped = analyze_new_media_file(
            args.media_files[0],
            label=args.label,
            add_to_training=args.add_to_training,
            whisper_model=args.whisper_model,
        )
        if args.export:
            export_analysis_report(wrapped, args.export)
        sys.exit(0 if wrapped["raw_result"].get("success") else 1)

    results = analyze_multiple_media_files(
        args.media_files,
        label=args.label,
        add_to_training=args.add_to_training,
        whisper_model=args.whisper_model,
    )
    if args.export:
        with open(args.export, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Report saved: {args.export}")
    sys.exit(0 if all(item.get("success") for item in results) else 1)


if __name__ == "__main__":
    main()
