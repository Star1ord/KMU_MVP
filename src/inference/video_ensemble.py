"""
Модуль для обработки видео и ансамблевого предсказания.

Интегрирует видео пайплайн с существующим аудио+NLP пайплайном.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
import numpy as np

# Добавляем путь к video_integration (legacy support)
VIDEO_INTEGRATION_DIR = Path(__file__).parent.parent.parent / 'video_integration'
sys.path.insert(0, str(VIDEO_INTEGRATION_DIR))

VIDEO_AVAILABLE = False
IMPORT_ERROR = None

# First, try to use the project's video_integration modules + mediapipe if present
try:
    import mediapipe as mp  # optional dependency used by some analyzers
    if not hasattr(mp, 'solutions'):
        raise ImportError("mediapipe installed but 'solutions' attribute not found. Try: pip install --upgrade mediapipe")

    from process_video import process_video_file, VideoAnalyzer  # type: ignore
    from use_video_model import VideoModelPredictor  # type: ignore

    VIDEO_AVAILABLE = True
    IMPORT_ERROR = None
except Exception as e:
    # If that fails, provide a lightweight fallback using the existing CV pipeline
    try:
        from src.pipeline.cv_pipeline.extract_visual_features import process_video_file as _process_video_file

        def process_video_file(*args, **kwargs):
            return _process_video_file(*args, **kwargs)

        class VideoModelPredictor:
            """Lightweight predictor that loads a joblib model and aggregates CSV features."""
            def __init__(self, model_path: Optional[str] = None, threshold: float = 0.6):
                possible_paths = [
                    Path('models/cv/model.joblib'),
                    Path('models/cv_audio/model.joblib'),
                    Path('models/video_model.joblib')
                ]
                mpath = Path(model_path) if model_path else None
                if (mpath is None) or (not mpath.exists()):
                    mpath = None
                    for p in possible_paths:
                        if p.exists():
                            mpath = p
                            break
                if (mpath is None) or (not mpath.exists()):
                    raise FileNotFoundError(f"Video model not found, checked: {possible_paths}")

                raw_model = joblib.load(mpath)

                # Спец-кейс: XGBoost модель, сохранённая как "сырой" dict с model_bytes.
                # Ожидаем структуру:
                # {
                #   "model_bytes": <bytes>,
                #   "feature_columns": [...],
                #   "params": {...},
                #   "best_round": int,
                # }
                if isinstance(raw_model, dict) and "model_bytes" in raw_model and "feature_columns" in raw_model:
                    try:
                        import xgboost as xgb
                    except Exception as e:
                        raise ImportError(
                            "Video model is an XGBoost snapshot (model_bytes), "
                            "but xgboost is not installed. "
                            "Install it, for example: pip install xgboost==3.1.2"
                        ) from e

                    model_bytes = raw_model["model_bytes"]
                    feature_columns = raw_model["feature_columns"]

                    if not isinstance(feature_columns, (list, tuple)) or not feature_columns:
                        raise ValueError(
                            f"Invalid feature_columns in video model: {feature_columns!r}"
                        )

                    booster = xgb.Booster()
                    # load_model в XGBoost 3.x умеет принимать bytes/bytearray
                    booster.load_model(bytearray(model_bytes))

                    class _XGBBinaryWrapper:
                        def __init__(self, booster, feature_cols):
                            self.booster = booster
                            self.feature_cols = list(feature_cols)

                        def predict_proba(self, X):
                            # X: numpy array shape (n_samples, n_features)
                            import xgboost as _xgb
                            dmat = _xgb.DMatrix(X, feature_names=self.feature_cols)
                            probs = self.booster.predict(dmat).reshape(-1, 1)
                            # Возвращаем как в sklearn: [P(class=0), P(class=1)]
                            probs = np.clip(probs, 1e-7, 1 - 1e-7)
                            return np.hstack([1.0 - probs, probs])

                    self.model = _XGBBinaryWrapper(booster, feature_columns)

                else:
                    # Общий случай: разворачиваем вложенные dict до estimator'а с predict_proba.
                    def _unwrap_model(obj, max_depth: int = 5):
                        current = obj
                        for _ in range(max_depth):
                            # Если это уже estimator с predict_proba — используем его
                            if hasattr(current, "predict_proba"):
                                return current
                            # Если это dict — пытаемся найти внутри модель
                            if isinstance(current, dict):
                                # 1) Прямой ключ 'model'
                                if "model" in current:
                                    current = current["model"]
                                    continue
                                # 2) Ищем первое значение с predict_proba или вложенный dict
                                next_candidate = None
                                for v in current.values():
                                    if hasattr(v, "predict_proba"):
                                        next_candidate = v
                                        break
                                    if isinstance(v, dict):
                                        next_candidate = v
                                if next_candidate is None:
                                    break
                                current = next_candidate
                                continue
                            # Не dict и без predict_proba — дальше разворачивать нечего
                            break
                        return current

                    self.model = _unwrap_model(raw_model)

                if not hasattr(self.model, "predict_proba"):
                    # Отдадим максимум отладочной инфы один раз, чтобы не гонять пайплайн снова
                    raise TypeError(
                        f"Loaded video model object has no predict_proba: "
                        f"type={type(self.model)}, "
                        f"raw_type={type(raw_model)}, "
                        f"raw_keys={list(raw_model.keys()) if isinstance(raw_model, dict) else 'n/a'}"
                    )

                self.threshold = threshold

            def predict_from_csv(self, df: pd.DataFrame):
                # If model wrapper provides expected feature order, use it.
                if hasattr(self.model, "feature_cols") and getattr(self.model, "feature_cols"):
                    expected_cols = list(getattr(self.model, "feature_cols"))
                    missing = [c for c in expected_cols if c not in df.columns]
                    if missing:
                        raise ValueError(
                            f"Missing required feature columns: {missing}. "
                            f"Available numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}"
                        )
                    X = df[expected_cols].mean(axis=0).values.reshape(1, -1)
                else:
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if not num_cols:
                        raise ValueError("No numeric features in video CSV")
                    X = df[num_cols].mean(axis=0).values.reshape(1, -1)
                prob = float(self.model.predict_proba(X)[0, 1])
                pred = 1 if prob >= self.threshold else 0
                return prob, pred

        VIDEO_AVAILABLE = True
        IMPORT_ERROR = None
    except Exception as fallback_e:
        VIDEO_AVAILABLE = False
        IMPORT_ERROR = f"Video processing import/fallback failed: {e}; {fallback_e}"


def process_video_for_prediction(
    video_path: str,
    output_dir: Optional[str] = None,
    sample_every: int = 3,
    use_emotions: Optional[bool] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Обрабатывает видео и создает CSV с видео фичами.
    
    Args:
        video_path: Путь к видео файлу
        output_dir: Директория для сохранения CSV (если None - временная)
        sample_every: Сэмплировать каждый N-й кадр
        use_emotions: Использовать эмоции (None = автоопределение)
        
    Returns:
        (csv_path, error_message): Путь к CSV файлу или None и сообщение об ошибке
    """
    if not VIDEO_AVAILABLE:
        return None, f"Video processing not available: {IMPORT_ERROR}"
    
    video_file = Path(video_path)
    if not video_file.exists():
        return None, f"Video file not found: {video_path}"
    
    try:
        # Определяем директорию для вывода
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='video_features_')
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        output_dir_path = Path(output_dir)
        
        # Имя выходного CSV файла
        base_name = video_file.stem
        csv_path = output_dir_path / f"{base_name}.csv"
        
        # Обрабатываем видео
        process_video_file(
            video_path=str(video_file),
            output_csv=str(csv_path),
            sample_every=sample_every,
            render_overlay=False,
            use_emotions=use_emotions,
            event_log_path=None
        )
        
        if csv_path.exists():
            return str(csv_path), None
        else:
            return None, "CSV file was not created"
            
    except Exception as e:
        return None, f"Error processing video: {str(e)}"


def predict_with_video_model(
    video_csv_path: str,
    model_path: Optional[str] = None,
    threshold: float = 0.6
) -> Dict:
    """
    Делает предсказание через видео модель.
    
    Args:
        video_csv_path: Путь к CSV файлу с видео фичами
        model_path: Путь к модели (по умолчанию video_integration/video_model.joblib)
        threshold: Порог классификации
        
    Returns:
        Словарь с результатами:
        - success: bool
        - probability: float
        - prediction: int (0 or 1)
        - risk_level: str
        - error: str (если ошибка)
    """
    if not VIDEO_AVAILABLE:
        return {
            'success': False,
            'error': f"Video model not available: {IMPORT_ERROR}"
        }
    
    result = {
        'success': False,
        'probability': None,
        'prediction': None,
        'risk_level': None,
        'error': None
    }
    
    try:
        # Определяем путь к модели
        if model_path is None:
            # Пробуем несколько возможных путей (including common model locations)
            possible_paths = [
                VIDEO_INTEGRATION_DIR / 'video_model.joblib',
                Path(__file__).parent.parent.parent / 'video_integration' / 'video_model.joblib',
                Path('video_integration') / 'video_model.joblib',
                Path('models') / 'cv' / 'model.joblib',
                Path('models') / 'cv_audio' / 'model.joblib',
                Path('models') / 'video_model.joblib',
            ]
            
            model_path = None
            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break
            
            if model_path is None:
                result['error'] = f"Video model not found. Checked: {[str(p) for p in possible_paths]}"
                return result
        
        model_path = Path(model_path)
        if not model_path.exists():
            result['error'] = f"Video model not found: {model_path}"
            return result
        
        # Загружаем CSV
        video_df = pd.read_csv(video_csv_path)
        if video_df.empty:
            result['error'] = "Video CSV is empty"
            return result
        
        # Инициализируем предсказатель
        predictor = VideoModelPredictor(
            model_path=str(model_path),
            threshold=threshold
        )
        
        # Делаем предсказание
        probability, prediction = predictor.predict_from_csv(video_df)
        
        # Определяем уровень риска
        if probability >= 0.7:
            risk_level = 'high'
        elif probability >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        result['success'] = True
        result['probability'] = float(probability)
        result['prediction'] = int(prediction)
        result['risk_level'] = risk_level
        
        return result
        
    except Exception as e:
        result['error'] = f"Error in video prediction: {str(e)}"
        return result


def create_ensemble_prediction(
    audio_nlp_result: Dict,
    video_result: Dict
) -> Dict:
    """
    Создает ансамблевое предсказание из результатов аудио+NLP и видео моделей.
    
    Args:
        audio_nlp_result: Результат от аудио+NLP модели
            - risk_score: float (вероятность)
            - prediction: int (0 or 1)
        video_result: Результат от видео модели
            - probability: float
            - prediction: int (0 or 1)
            
    Returns:
        Словарь с ансамблевыми результатами:
        - audio_nlp_score: float
        - video_score: float
        - ensemble_score: float (среднее)
        - ensemble_prediction: int
        - agreement: bool (согласны ли модели)
        - risk_level: str
    """
    audio_score = audio_nlp_result.get('risk_score', 0.0)
    video_score = video_result.get('probability', 0.0)
    
    # Среднее вероятностей (простое ансамблевое предсказание)
    ensemble_score = (audio_score + video_score) / 2.0
    
    # Порог для ансамбля (можно настроить)
    ensemble_threshold = 0.5
    ensemble_prediction = 1 if ensemble_score >= ensemble_threshold else 0
    
    # Проверяем согласованность моделей
    audio_pred = audio_nlp_result.get('prediction', 0)
    video_pred = video_result.get('prediction', 0)
    agreement = (audio_pred == video_pred)
    
    # Определяем уровень риска для ансамбля
    if ensemble_score >= 0.7:
        risk_level = 'high'
    elif ensemble_score >= 0.4:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    return {
        'audio_nlp_score': float(audio_score),
        'video_score': float(video_score),
        'ensemble_score': float(ensemble_score),
        'ensemble_prediction': int(ensemble_prediction),
        'agreement': bool(agreement),
        'risk_level': risk_level,
        'audio_nlp_prediction': int(audio_pred),
        'video_prediction': int(video_pred)
    }



