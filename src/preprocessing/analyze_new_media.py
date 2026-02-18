#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze new media file with segment-level predictions

This module:
1. Processes media through pipeline
2. Creates temporary CSV with correct format
3. Runs segment model for interpretability
4. Returns predictions with timestamps
5. Optionally adds to training dataset
6. Cleans up temporary files
"""

import os
import sys
import tempfile
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json

# Ensure repository root and `src` package are on sys.path so the module
# can be run directly with `python src/preprocessing/analyze_new_media.py`.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import DATA_DIR, MODELS_DIR
from src.preprocessing.process_new_media import process_new_media

# Video ensemble will be imported lazily (defer heavy imports until needed)
VIDEO_ENSEMBLE_AVAILABLE = None
VIDEO_IMPORT_ERROR = None


def analyze_new_media_file(
    media_path: str,
    label: Optional[int] = None,
    add_to_training: bool = False,
    whisper_model: str = 'medium',
    skip_transcription: bool = False,
    skip_video: bool = False,
) -> Dict:
    """
    Analyze a single media file and return predictions
    
    Args:
        media_path: Path to video/audio file
        label: Label for training (0=control, 1=risk). None for inference only
        add_to_training: Whether to add to training dataset
        whisper_model: WhisperX model to use
        
    Returns:
        Dictionary with:
        - success: bool
        - session_id: str
        - prediction: int (0 or 1)
        - risk_score: float (0-1)
        - risk_level: str ('low', 'medium', 'high')
        - segments: list of segment predictions with timestamps
        - features_csv: path to features CSV (temporary)
        - error: str (if failed)
    """
    
    result = {
        'success': False,
        'session_id': None,
        'prediction': None,
        'risk_score': None,
        'risk_level': None,
        'segments': [],
        'features_csv': None,
        'error': None,
        'video_result': None,
        'ensemble_result': None
    }
    
    media_file = Path(media_path)
    if not media_file.exists():
        result['error'] = f"File not found: {media_path}"
        return result
    
    session_id = media_file.stem
    result['session_id'] = session_id
    
    # Сохраняем оригинальный session_id для отладки
    original_session_id = session_id
    
    # Store original media path for audio playback
    original_media_path = str(media_file)
    
    # Create temporary directory for processing
    temp_dir = Path(tempfile.mkdtemp(prefix='analyze_'))
    temp_media = temp_dir / media_file.name
    
    try:
        # Copy file to temp
        shutil.copy2(media_path, temp_media)
        
        # Process through pipeline
        print(f"\n{'='*80}")
        print(f"Analyzing: {session_id}")
        print(f"{'='*80}")
        
        if label is None:
            label = 0  # Default for inference
        
        success = process_new_media(
            media_files=[str(temp_media)],
            label=label,
            whisper_model=whisper_model,
            whisper_device='cpu',
            skip_pipeline=False,
            add_to_training=add_to_training,
            skip_transcription=skip_transcription,
        )
        
        if not success:
            # Don't abort here — try to proceed with a minimal fallback so the model can still run
            print("warning: pipeline reported failure; attempting fallback to generate minimal features")
            result['pipeline_failed'] = True
        
        # Possible locations for merged_features.csv (try each)
        possible_paths = [
            os.path.join(DATA_DIR, 'merged_features.csv'),
            os.path.join('data', 'ml', 'merged_features.csv'),
            os.path.join('data', 'processed', 'features', 'merged_features.csv'),
        ]

        merged_features_path = None
        for path in possible_paths:
            if os.path.exists(path):
                merged_features_path = path
                break
        
        if not merged_features_path:
            # Fallback: create a minimal features CSV so the model can still run
            fallback_dir = Path('data/processed') / 'features'
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_dir / 'merged_features.csv'

            print(f"Features not found. Creating fallback minimal features at {fallback_path}")

            # Build a single-row minimal dataframe for this session
            import pandas as _pd
            minimal = _pd.DataFrame([
                {
                    'file_id': session_id,
                    'label': label if label is not None else 0,
                    'segment_id': f"{session_id}_segment_0000",
                    'start': 0.0,
                    'end': 0.0,
                    'duration': 0.0,
                    'text': ''
                }
            ])
            minimal.to_csv(fallback_path, index=False, encoding='utf-8')
            merged_features_path = str(fallback_path)
            print(f"Fallback features saved to: {merged_features_path}")
        
        print(f"Loading features from: {merged_features_path}")
        
        # Инициализируем переменные для видео обработки
        video_result = None
        cv_audio_result = None
        ensemble_result = None

        # Determine whether input file is video (used for CV/AV pipelines)
        media_ext = media_file.suffix.lower()
        is_video = media_ext in ['.mp4', '.mov', '.mkv', '.avi', '.webm']

        # Try to import video ensemble lazily to avoid heavy imports at module import time
        VIDEO_ENSEMBLE_AVAILABLE = False
        VIDEO_IMPORT_ERROR = None
        if not skip_video:
            try:
                from inference.video_ensemble import (
                    process_video_for_prediction,
                    predict_with_video_model,
                    create_ensemble_prediction,
                )
                VIDEO_ENSEMBLE_AVAILABLE = True
            except Exception as e:
                VIDEO_ENSEMBLE_AVAILABLE = False
                VIDEO_IMPORT_ERROR = str(e)

        # Try to import CV+Audio pipeline (we keep this available even if video model is skipped)
        try:
            from src.pipeline.cv_audio_pipeline import predict_cv_audio as predict_cv_audio_func
            CV_AUDIO_AVAILABLE = True
        except Exception as e:
            CV_AUDIO_AVAILABLE = False
            CV_AUDIO_IMPORT_ERROR = str(e)

        # Video processing (only if not skipped and video ensemble available)
        if not skip_video and VIDEO_ENSEMBLE_AVAILABLE and is_video:
            print(f"\n{'='*80}")
            print("VIDEO PROCESSING (independent from audio+NLP pipeline)")
            print(f"{'='*80}")

            try:
                # Используем оригинальный путь к файлу (не временный)
                video_csv_path, video_error = process_video_for_prediction(
                    video_path=str(media_file),
                    output_dir=None,
                    sample_every=3,
                    use_emotions=None
                )

                if video_csv_path and not video_error:
                    print(f"Video features CSV created: {video_csv_path}")

                    video_pred_result = predict_with_video_model(
                        video_csv_path=video_csv_path,
                        model_path=None,
                        threshold=0.6
                    )

                    if video_pred_result['success']:
                        video_result = video_pred_result
                        print(f"\n[Video Model Prediction]")
                        print(f"  Video score: {video_result['probability']:.3f}")
                        print(f"  Prediction: {'RISK' if video_result['prediction'] == 1 else 'CONTROL'}")
                        print(f"  Risk level: {video_result['risk_level']}")
                    else:
                        print(f"Video prediction failed: {video_pred_result.get('error', 'Unknown error')}")
                else:
                    print(f"Video processing failed: {video_error}")
            except Exception as video_e:
                print(f"Error in video processing: {video_e}")
                import traceback
                traceback.print_exc()

        # CV+Audio processing (if available and input is video) — keep this even when skip_video=True
        if CV_AUDIO_AVAILABLE and is_video:
            print(f"\n{'='*80}")
            print("CV+AUDIO PROCESSING")
            print(f"{'='*80}")

            try:
                cv_audio_pred_result = predict_cv_audio_func(
                    video_path=str(media_file),
                    model_path=None,
                    threshold=0.5,
                    sample_rate=1.0,
                )

                if cv_audio_pred_result.get('success'):
                    cv_audio_result = cv_audio_pred_result
                    print(f"\n[CV+Audio Model Prediction]")
                    print(f"  CV+Audio score: {cv_audio_result['probability']:.3f}")
                    print(f"  Prediction: {'RISK' if cv_audio_result['prediction'] == 1 else 'CONTROL'}")
                    print(f"  Risk level: {cv_audio_result['risk_level']}")
                else:
                    print(f"CV+Audio prediction failed: {cv_audio_pred_result.get('error', 'Unknown error')}")
            except Exception as cv_audio_e:
                print(f"Error in CV+Audio processing: {cv_audio_e}")
                import traceback
                traceback.print_exc()
        
        # Load features for this session
        df = pd.read_csv(merged_features_path, low_memory=False)
        print(f"Loaded {len(df)} total segments from merged_features.csv")
        
        # Получаем все уникальные file_id для отладки
        available_file_ids = df['file_id'].unique() if 'file_id' in df.columns else []
        
        # Пробуем разные варианты поиска session_id
        session_data = None
        found_session_id = None
        
        # Варианты для поиска (в порядке приоритета)
        search_variants = [
            session_id,  # Оригинальное имя
            f"{label}_{session_id}",  # С префиксом label (0_ или 1_)
            session_id.replace('0_', '').replace('1_', ''),  # Без префикса
            f"0_{session_id}",  # С префиксом 0_
            f"1_{session_id}",  # С префиксом 1_
        ]
        
        # Также пробуем частичное совпадение (если session_id содержится в file_id)
        for variant in search_variants:
            session_data = df[df['file_id'] == variant].copy()
            if len(session_data) > 0:
                found_session_id = variant
                break
        
        # Если не нашли точное совпадение, пробуем частичное
        if session_data is None or len(session_data) == 0:
            for file_id in available_file_ids:
                if session_id in str(file_id) or str(file_id) in session_id:
                    session_data = df[df['file_id'] == file_id].copy()
                    if len(session_data) > 0:
                        found_session_id = file_id
                        print(f"Found partial match: {file_id} for session {session_id}")
                        break
        
        # Если все еще не нашли, выводим отладочную информацию
        if session_data is None or len(session_data) == 0:
            # Если есть видео результат, возвращаем его
            if video_result and video_result['success']:
                result['success'] = True
                result['prediction'] = video_result['prediction']
                result['risk_score'] = video_result['probability']
                result['risk_level'] = video_result['risk_level']
                result['video_result'] = video_result
                result['error'] = f"Audio+NLP pipeline failed (no segments found), but video analysis completed"
                print(f"\n⚠️  Audio+NLP pipeline failed, but video analysis succeeded")
                return result
            
            # Нет ни аудио+NLP, ни видео результатов
            error_msg = f"No segments found for session '{session_id}' (label={label})\n"
            error_msg += f"Tried variants: {search_variants}\n"
            error_msg += f"Available file_ids in dataset: {len(available_file_ids)} unique IDs\n"
            if len(available_file_ids) > 0:
                sample_ids = list(available_file_ids)[:10]
                error_msg += f"Sample file_ids: {sample_ids}\n"
            error_msg += f"\nPossible reasons:\n"
            error_msg += f"1. Pipeline processing may have failed\n"
            error_msg += f"2. File ID format mismatch\n"
            error_msg += f"3. Segments were not created (check pipeline logs)\n"
            if video_result:
                error_msg += f"\nNote: Video processing attempted but failed: {video_result.get('error', 'Unknown')}"
            
            result['error'] = error_msg
            result['video_result'] = video_result
            print(f"\n{'='*80}")
            print("ERROR: Session not found")
            print(f"{'='*80}")
            print(error_msg)
            return result
        
        # Обновляем session_id на найденный
        if found_session_id:
            session_id = found_session_id
            result['session_id'] = session_id
            print(f"Found session data with file_id: {session_id} ({len(session_data)} segments)")
        
        result['features_csv'] = merged_features_path
        
        # Use Late Fusion V3 Clean for FINAL prediction (production model)
        # Segment model is only for visualization/interpretability
        try:
            import joblib
            import numpy as np
            
            # Load Late Fusion V3 Clean model (try multiple candidate locations)
            candidate_paths = [
                os.path.join(MODELS_DIR, 'late_fusion_v3_clean.pkl'),
                os.path.join(str(ROOT), 'models', 'nlp', 'late_fusion_v3_clean.pkl'),
                os.path.join(str(ROOT), 'models', 'late_fusion_v3_clean.pkl'),
            ]
            model_path = None
            for p in candidate_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            if model_path is None:
                raise FileNotFoundError(f"Production model not found. Checked: {candidate_paths}")

            model = joblib.load(model_path)
            
            # Build session-level features (aggregate segments)
            # Need to match multi_session_df format: audio_F0..._mean, audio_F0..._std, etc.
            # Model expects aggregated features (mean, std, median, min, max)
            audio_cols = model['audio_feature_names']
            
            # Extract base feature names from model columns
            # Model format: audio_F0semitoneFrom27.5Hz_sma3nz_amean_mean
            # Segment format: F0semitoneFrom27.5Hz_sma3nz_amean (no prefix, no suffix)
            base_features = {}
            for col in audio_cols:
                # Pattern: audio_F0..._mean -> base: F0...
                match = re.match(r'audio_(.+)_(mean|std|median|min|max)$', col)
                if match:
                    base_name = match.group(1)
                    stat = match.group(2)
                    if base_name not in base_features:
                        base_features[base_name] = []
                    base_features[base_name].append((stat, col))
            
            # Aggregate segment features
            audio_features = {}
            exclude_cols = ['file_id', 'segment_id', 'start', 'end', 'duration', 
                           'asr_conf_mean', 'asr_conf_std', 'word_count', 
                           'segment_path', 'text', 'label']
            
            # Segment data has columns WITHOUT audio_ prefix (e.g., F0semitoneFrom27.5Hz_sma3nz_amean)
            segment_audio_cols = [c for c in session_data.columns 
                                 if c not in exclude_cols and not c.startswith('text_')]
            
            # Aggregate each base feature
            for base_name, stats in base_features.items():
                # Find matching column in segment data (exact match, no prefix)
                matching_col = base_name if base_name in session_data.columns else None
                
                if matching_col and matching_col in session_data.columns:
                    values = session_data[matching_col].values
                    values = values[~pd.isna(values)]
                    
                    if len(values) > 0:
                        for stat, col_name in stats:
                            if stat == 'mean':
                                audio_features[col_name] = float(np.mean(values))
                            elif stat == 'std':
                                audio_features[col_name] = float(np.std(values)) if len(values) > 1 else 0.0
                            elif stat == 'median':
                                audio_features[col_name] = float(np.median(values))
                            elif stat == 'min':
                                audio_features[col_name] = float(np.min(values))
                            elif stat == 'max':
                                audio_features[col_name] = float(np.max(values))
                    else:
                        # All NaN - set to 0
                        for stat, col_name in stats:
                            audio_features[col_name] = 0.0
                else:
                    # Column not found - set to 0
                    for stat, col_name in stats:
                        audio_features[col_name] = 0.0
            
            # Fill any missing columns
            for col in audio_cols:
                if col not in audio_features:
                    audio_features[col] = 0.0
            
            print(f"\n[Aggregation] Base features: {len(base_features)}, Aggregated: {len(audio_features)}")
            
            # Get full text
            full_text = ' '.join(session_data['text'].fillna('').astype(str).tolist())
            
            # Predict with Late Fusion V3
            text_model = model['text_model']
            audio_model = model['audio_model']
            meta_model = model['meta_model']
            threshold = model['threshold']
            
            # Text prediction
            X_tfidf = text_model['vectorizer'].transform([full_text])
            p_text = text_model['classifier'].predict_proba(X_tfidf)[0, 1]
            
            # Audio prediction
            X_audio = np.array([[audio_features[col] for col in audio_cols]])
            X_audio = np.nan_to_num(X_audio, nan=0.0, posinf=0.0, neginf=0.0)
            p_audio = audio_model.predict_proba(X_audio)[0, 1]
            
            # Meta prediction
            X_meta = np.array([[p_text, p_audio]])
            p_final = meta_model.predict_proba(X_meta)[0, 1]
            
            # Final prediction
            final_prediction = 1 if p_final >= threshold else 0
            
            print(f"\n[Late Fusion V3 Clean Prediction]")
            print(f"  Text score: {p_text:.3f}")
            print(f"  Audio score: {p_audio:.3f}")
            print(f"  Final score: {p_final:.3f}")
            print(f"  Threshold: {threshold:.3f}")
            print(f"  Prediction: {'RISK' if final_prediction == 1 else 'CONTROL'}")
            
            result['success'] = True
            result['prediction'] = final_prediction
            result['risk_score'] = float(p_final)
            result['risk_level'] = 'high' if p_final >= 0.6 else 'medium' if p_final >= 0.3 else 'low'
            
            # Если видео уже обработано ранее, создаем ансамблевое предсказание
            # Собираем все доступные результаты моделей
            available_results = []
            
            # Audio+NLP результат (основной)
            if result.get('success') and result.get('risk_score') is not None:
                available_results.append({
                    'name': 'nlp',
                    'score': result['risk_score'],
                    'prediction': result['prediction']
                })
            
            # Video результат
            if video_result and video_result.get('success'):
                available_results.append({
                    'name': 'video',
                    'score': video_result['probability'],
                    'prediction': video_result['prediction']
                })
            
            # CV+Audio результат
            if cv_audio_result and cv_audio_result.get('success'):
                available_results.append({
                    'name': 'cv_audio',
                    'score': cv_audio_result['probability'],
                    'prediction': cv_audio_result['prediction']
                })
            
            # Создаём ensemble, если есть хотя бы 2 модели
            if len(available_results) >= 2:
                scores = [r['score'] for r in available_results]
                predictions = [r['prediction'] for r in available_results]
                
                # Простое среднее вероятностей
                ensemble_score = np.mean(scores)
                ensemble_prediction = 1 if ensemble_score >= 0.5 else 0
                
                # Проверяем согласованность
                agreement = len(set(predictions)) == 1
                
                ensemble_result = {
                    'ensemble_score': float(ensemble_score),
                    'ensemble_prediction': int(ensemble_prediction),
                    'ensemble_prediction_label': 'experimental' if ensemble_prediction == 1 else 'control',
                    'models_used': [r['name'] for r in available_results],
                    'individual_scores': {r['name']: r['score'] for r in available_results},
                    'individual_predictions': {r['name']: r['prediction'] for r in available_results},
                    'agreement': agreement,
                    'risk_level': 'high' if ensemble_score >= 0.7 else 'medium' if ensemble_score >= 0.4 else 'low'
                }
                
                print(f"\n[Ensemble Prediction]")
                for r in available_results:
                    print(f"  {r['name']} score: {r['score']:.3f}")
                print(f"  Ensemble score: {ensemble_result['ensemble_score']:.3f}")
                print(f"  Ensemble prediction: {ensemble_result['ensemble_prediction_label'].upper()}")
                print(f"  Models agreement: {'Yes' if agreement else 'No'}")
            elif len(available_results) == 1:
                # Только одна модель - используем её результат как ensemble
                r = available_results[0]
                ensemble_result = {
                    'ensemble_score': float(r['score']),
                    'ensemble_prediction': int(r['prediction']),
                    'ensemble_prediction_label': 'experimental' if r['prediction'] == 1 else 'control',
                    'models_used': [r['name']],
                    'individual_scores': {r['name']: r['score']},
                    'individual_predictions': {r['name']: r['prediction']},
                    'agreement': True,
                    'risk_level': 'high' if r['score'] >= 0.7 else 'medium' if r['score'] >= 0.4 else 'low'
                }
            
            result['video_result'] = video_result
            result['cv_audio_result'] = cv_audio_result
            result['ensemble_result'] = ensemble_result
            
            # Also get segment-level analysis for visualization (using segment model)
            try:
                from inference.segment_analyzer import SegmentAnalyzer
                analyzer = SegmentAnalyzer()
                analysis = analyzer.analyze_session(session_data, session_id)
            
                # Get all feature names
                feature_names = analyzer.feature_names
                audio_features = analyzer.audio_features
                text_features = analyzer.text_features
                
                # Format segments for visualization with all features
                for i, seg in enumerate(analysis.segments):
                    # Get corresponding row from session_data
                    seg_row = session_data.iloc[i] if i < len(session_data) else None
                    
                    # Extract all feature values for this segment
                    all_features = {}
                    if seg_row is not None:
                        for feat_name in feature_names:
                            if feat_name in session_data.columns:
                                feat_value = seg_row[feat_name]
                                if pd.notna(feat_value):
                                    all_features[feat_name] = {
                                        'value': float(feat_value),
                                        'type': 'audio' if feat_name in audio_features else 'text'
                                    }
                                else:
                                    all_features[feat_name] = {
                                        'value': 0.0,
                                        'type': 'audio' if feat_name in audio_features else 'text'
                                    }
                    
                    result['segments'].append({
                        'segment_id': seg.segment_id,
                        'start_time': seg.start_time,
                        'end_time': seg.end_time,
                        'duration': seg.duration,
                        'text': seg.text,
                        'risk_score': seg.risk_score,
                        'risk_level': seg.risk_level,
                        'top_features': seg.top_features[:5],  # Top 5 features for quick view
                        'all_features': all_features  # All features for detailed visualization
                    })
                
                # Store feature metadata
                result['feature_names'] = feature_names
                result['audio_features'] = audio_features
                result['text_features'] = text_features
                
            except Exception as seg_error:
                # If segment model fails, still return main prediction but no segments
                print(f"Warning: Segment analysis failed: {seg_error}")
                result['segments'] = []
            
            # Try to find audio file path
            audio_path = None
            
            # First, try to use original media file if it's audio
            if original_media_path and Path(original_media_path).exists():
                ext = Path(original_media_path).suffix.lower()
                if ext in ['.wav', '.mp3', '.m4a']:
                    audio_path = original_media_path
            
            # Try processed audio location
            if audio_path is None:
                audio_base_dir = Path('data/raw/audio_wav')
                for label_dir in [0, 1]:
                    potential_path = audio_base_dir / str(label_dir) / f"{session_id}.wav"
                    if potential_path.exists():
                        audio_path = str(potential_path)
                        break
            
            # Check if audio_path column exists in session_data
            if audio_path is None and 'audio_path' in session_data.columns:
                audio_paths = session_data['audio_path'].dropna().unique()
                if len(audio_paths) > 0:
                    potential_path = audio_paths[0]
                    if Path(potential_path).exists():
                        audio_path = potential_path
            
            result['audio_path'] = audio_path
            
            print(f"\n{'='*80}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*80}")
            print(f"Session: {session_id}")
            print(f"Prediction: {'RISK' if result['prediction'] == 1 else 'CONTROL'}")
            print(f"Risk Score: {result['risk_score']*100:.1f}%")
            print(f"Segments: {len(result['segments'])}")
            print(f"High-risk segments: {sum(1 for s in result['segments'] if s['risk_level'] == 'high')}")
            
        except FileNotFoundError as e:
            result['error'] = f"Model not found: {str(e)}"
            import traceback
            traceback.print_exc()
            return result
        except Exception as e:
            result['error'] = f"Error during analysis: {str(e)}"
            import traceback
            traceback.print_exc()
            return result
        
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary files: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temp dir: {e}")
    
    # Build a cleaned, structured output for easier consumption while
    # keeping the original `result` for backward compatibility.
    try:
        ensemble = result.get('ensemble_result') if result.get('ensemble_result') else None

        # Per-model summaries
        models_summary = {}
        # NLP / audio model (nlp) is represented by top-level prediction/risk_score
        models_summary['nlp'] = {
            'success': bool(result.get('success', False)),
            'prediction': int(result.get('prediction')) if result.get('prediction') is not None else None,
            'risk_level': result.get('risk_level'),
            'probability': float(result.get('risk_score')) if result.get('risk_score') is not None else None
        }

        # CV+Audio model
        if result.get('cv_audio_result'):
            cv = result['cv_audio_result']
            models_summary['cv_audio'] = {
                'success': bool(cv.get('success', False)),
                'prediction': int(cv.get('prediction')) if cv.get('prediction') is not None else None,
                'risk_level': cv.get('risk_level'),
                'probability': float(cv.get('probability')) if cv.get('probability') is not None else None,
                'error': cv.get('error')
            }
        else:
            models_summary['cv_audio'] = None

        # Overall / ensemble summary
        overall = {
            'success': bool(result.get('success', False)),
            'prediction': int(result.get('prediction')) if result.get('prediction') is not None else None,
            'risk_level': result.get('risk_level'),
            'risk_score': float(result.get('risk_score')) if result.get('risk_score') is not None else None,
            'ensemble': None
        }

        if ensemble:
            overall['ensemble'] = {
                'ensemble_score': float(ensemble.get('ensemble_score')),
                'ensemble_prediction': int(ensemble.get('ensemble_prediction')),
                'ensemble_prediction_label': ensemble.get('ensemble_prediction_label'),
                'models_used': ensemble.get('models_used'),
                'individual_scores': ensemble.get('individual_scores'),
                'individual_predictions': ensemble.get('individual_predictions'),
                'agreement': bool(ensemble.get('agreement')),
                'risk_level': ensemble.get('risk_level')
            }

        pretty = {
            'success': result.get('success', False),
            'session_id': result.get('session_id'),
            'overall': overall,
            'models': models_summary,
            'segments': result.get('segments', []),
            'features_csv': result.get('features_csv'),
            'audio_path': result.get('audio_path'),
            'error': result.get('error'),
        }

    except Exception:
        # If formatting fails for any reason, fall back to original result
        return result

    # Return the pretty formatted result while keeping the original raw data
    return {'formatted_result': pretty, 'raw_result': result}


def analyze_multiple_media_files(
    media_paths: List[str],
    label: Optional[int] = None,
    add_to_training: bool = False,
    whisper_model: str = 'medium'
) -> List[Dict]:
    """
    Analyze multiple media files
    
    Args:
        media_paths: List of paths to video/audio files
        label: Label for training (0=control, 1=risk)
        add_to_training: Whether to add to training dataset
        whisper_model: WhisperX model to use
        
    Returns:
        List of analysis results (one per file)
    """
    results = []
    
    for media_path in media_paths:
        print(f"\n{'#'*80}")
        print(f"Processing: {Path(media_path).name}")
        print(f"{'#'*80}")
        
        result = analyze_new_media_file(
            media_path,
            label=label,
            add_to_training=add_to_training,
            whisper_model=whisper_model
        )
        
        results.append(result)
        
        if not result['success']:
            print(f"ERROR: {result['error']}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    
    successful = [r for r in results if r['success']]
    if successful:
        risk_count = sum(1 for r in successful if r['prediction'] == 1)
        print(f"\nPredictions:")
        print(f"  RISK: {risk_count}")
        print(f"  CONTROL: {len(successful) - risk_count}")
    
    return results


def export_analysis_report(result: Dict, output_path: str):
    """Export analysis result to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze new media files with segment-level predictions')
    parser.add_argument('media_files', nargs='+', help='Media files to analyze')
    parser.add_argument('--label', type=int, choices=[0, 1], help='Label (0=control, 1=risk)')
    parser.add_argument('--add-to-training', action='store_true', help='Add to training dataset')
    parser.add_argument('--whisper-model', default='medium', choices=['medium', 'large'])
    parser.add_argument('--export', help='Export report to JSON file')
    
    args = parser.parse_args()
    
    if len(args.media_files) == 1:
        result = analyze_new_media_file(
            args.media_files[0],
            label=args.label,
            add_to_training=args.add_to_training,
            whisper_model=args.whisper_model
        )
        
        if args.export:
            export_analysis_report(result, args.export)
        
        sys.exit(0 if result['success'] else 1)
    else:
        results = analyze_multiple_media_files(
            args.media_files,
            label=args.label,
            add_to_training=args.add_to_training,
            whisper_model=args.whisper_model
        )
        
        if args.export:
            with open(args.export, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Report saved: {args.export}")
        
        success_count = sum(1 for r in results if r['success'])
        sys.exit(0 if success_count == len(results) else 1)


if __name__ == '__main__':
    main()

