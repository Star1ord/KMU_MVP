#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference API module - Core prediction logic extracted from app.py.
Reusable inference functions for both Streamlit UI and REST API.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR


class InferenceEngine:
    """Unified inference engine for model predictions"""
    
    def __init__(self):
        self.models_cache = {}
        self.metadata_cache = {}
    
    def load_model(self, model_type: str = 'late_fusion') -> Tuple[Any, Dict]:
        """Load model and metadata from disk with caching"""
        
        if model_type in self.models_cache:
            return self.models_cache[model_type], self.metadata_cache[model_type]
        
        model_paths = {
            'late_fusion': ('late_fusion_v3_clean.pkl', 'late_fusion_v3_clean_metadata.json'),
            'early_fusion_catboost': ('early_fusion_catboost.pkl', 'early_fusion_metadata.json'),
            'early_fusion_linear': ('early_fusion_linear_svc.pkl', 'early_fusion_metadata.json')
        }
        
        if model_type not in model_paths:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_file, metadata_file = model_paths[model_type]
        model_path = os.path.join(MODELS_DIR, model_file)
        metadata_path = os.path.join(RESULTS_DIR, metadata_file)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Cache
        self.models_cache[model_type] = model
        self.metadata_cache[model_type] = metadata
        
        return model, metadata
    
    def predict(
        self,
        model_type: str,
        session_data: Dict[str, Any],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Make prediction for a single session.
        
        Args:
            model_type: Type of model to use
            session_data: Dictionary with audio_features and text_features
            threshold: Custom decision threshold
        
        Returns:
            Prediction result dictionary
        """
        
        model, metadata = self.load_model(model_type)
        
        # Create DataFrame from features
        df = pd.DataFrame([{
            **session_data.get('audio_features', {}),
            **session_data.get('text_features', {})
        }])
        
        # Get feature columns
        audio_cols = metadata.get('audio_feature_names', [])
        text_cols = metadata.get('text_feature_names', [])
        
        # Handle missing features
        for col in audio_cols + text_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Prepare data
        if model_type == 'late_fusion':
            return self._predict_late_fusion(model, metadata, df, audio_cols, text_cols, threshold)
        else:
            return self._predict_early_fusion(model, metadata, df, audio_cols, text_cols, threshold)
    
    def _predict_late_fusion(
        self,
        model: Dict,
        metadata: Dict,
        df: pd.DataFrame,
        audio_cols: list,
        text_cols: list,
        threshold: Optional[float]
    ) -> Dict[str, Any]:
        """Late Fusion prediction (separate audio/text models + meta-learner)"""
        
        # Audio model
        audio_model = model.get('audio_model')
        X_audio = df[audio_cols].fillna(0).values
        X_audio = np.nan_to_num(X_audio, nan=0.0, posinf=0.0, neginf=0.0)
        p_audio = audio_model.predict_proba(X_audio)[:, 1] if audio_model else np.array([0.5])
        
        # Text model
        text_model = model.get('text_model')
        if text_model:
            tfidf = text_model.get('tfidf')
            classifier = text_model.get('classifier')
            # Assuming text data available - placeholder
            p_text = np.array([0.5])
        else:
            p_text = np.array([0.5])
        
        # Meta model
        meta_model = model.get('meta_model')
        X_meta = np.column_stack([p_text, p_audio])
        probabilities = meta_model.predict_proba(X_meta)[:, 1]
        
        probability = float(probabilities[0])
        
        # Use provided threshold or metadata
        if threshold is None:
            threshold = metadata.get('threshold', 0.5)
        
        decision = bool(probability >= threshold)
        
        # Feature importance (audio features)
        feature_importance = {}
        if audio_model and hasattr(audio_model, 'get_feature_importance'):
            try:
                importance = audio_model.get_feature_importance()
                top_indices = np.argsort(importance)[-5:][::-1]
                feature_importance = {
                    audio_cols[i]: float(importance[i]) 
                    for i in top_indices
                }
            except:
                pass
        
        return {
            'probability': probability,
            'decision': decision,
            'threshold': threshold,
            'confidence': max(probability, 1 - probability),
            'feature_importance': feature_importance
        }
    
    def _predict_early_fusion(
        self,
        model: Any,
        metadata: Dict,
        df: pd.DataFrame,
        audio_cols: list,
        text_cols: list,
        threshold: Optional[float]
    ) -> Dict[str, Any]:
        """Early Fusion prediction (combined features)"""
        
        # Handle both dict and direct model formats
        if isinstance(model, dict) and 'model' in model:
            actual_model = model['model']['model'] if isinstance(model['model'], dict) else model['model']
        else:
            actual_model = model
        
        # Prepare features
        X_audio = df[audio_cols].fillna(0).values if audio_cols else np.array([[]])
        X_text = df[text_cols].fillna(0).values if text_cols else np.array([[]])
        
        if X_audio.size > 0 and X_text.size > 0:
            X_combined = np.hstack([X_text, X_audio])
        elif X_audio.size > 0:
            X_combined = X_audio
        else:
            X_combined = X_text
        
        X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Prediction
        probabilities = actual_model.predict_proba(X_combined)[:, 1]
        probability = float(probabilities[0])
        
        # Threshold
        if threshold is None:
            threshold = metadata.get('threshold', 0.5)
        
        decision = bool(probability >= threshold)
        
        # Feature importance
        feature_importance = {}
        if hasattr(actual_model, 'get_feature_importance'):
            try:
                importance = actual_model.get_feature_importance()
                feature_names = text_cols + audio_cols if text_cols else audio_cols
                top_indices = np.argsort(importance)[-5:][::-1]
                feature_importance = {
                    feature_names[i]: float(importance[i])
                    for i in top_indices
                    if i < len(feature_names)
                }
            except:
                pass
        
        return {
            'probability': probability,
            'decision': decision,
            'threshold': threshold,
            'confidence': max(probability, 1 - probability),
            'feature_importance': feature_importance
        }
    
    def batch_predict(
        self,
        model_type: str,
        sessions: list,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make predictions for multiple sessions"""
        
        results = []
        for session in sessions:
            result = self.predict(model_type, session, threshold)
            results.append(result)
        
        # Summary
        probabilities = [r['probability'] for r in results]
        decisions = [r['decision'] for r in results]
        
        summary = {
            'total': len(results),
            'at_risk': sum(decisions),
            'safe': len(decisions) - sum(decisions),
            'mean_probability': float(np.mean(probabilities)),
            'std_probability': float(np.std(probabilities)),
            'min_probability': float(np.min(probabilities)),
            'max_probability': float(np.max(probabilities))
        }
        
        return {
            'results': results,
            'summary': summary
        }
    
    def get_models_info(self) -> Dict[str, Dict]:
        """Get information about available models"""
        
        models_info = {}
        
        for model_type in ['late_fusion', 'early_fusion_catboost', 'early_fusion_linear']:
            try:
                model, metadata = self.load_model(model_type)
                models_info[model_type] = {
                    'name': model_type.replace('_', ' ').title(),
                    'type': model_type,
                    'version': metadata.get('version', 'unknown'),
                    'metrics': {
                        'roc_auc': metadata.get('mean_roc_auc'),
                        'pr_auc': metadata.get('mean_pr_auc'),
                        'f1_score': metadata.get('mean_f1')
                    }
                }
            except Exception as e:
                models_info[model_type] = {
                    'name': model_type.replace('_', ' ').title(),
                    'type': model_type,
                    'error': str(e)
                }
        
        return models_info


# Global instance
_inference_engine = None

def get_inference_engine() -> InferenceEngine:
    """Get or create inference engine singleton"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine
