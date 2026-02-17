#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment Analyzer - Interpretable predictions for suicide risk detection

This module provides:
1. Segment-level predictions with risk scores
2. Identification of high-risk segments (triggers)
3. Timeline visualization of risk across the session
4. Feature contribution analysis per segment
"""

import numpy as np
import pandas as pd
import joblib
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import MODELS_DIR, DATA_DIR


@dataclass
class SegmentPrediction:
    """Single segment prediction with metadata"""
    segment_id: str
    start_time: float
    end_time: float
    duration: float
    text: str
    risk_score: float
    risk_level: str  # 'low', 'medium', 'high'
    top_features: List[Dict]  # Features that contributed most to this prediction


@dataclass
class SessionAnalysis:
    """Full session analysis result"""
    session_id: str
    overall_risk_score: float
    overall_risk_level: str
    prediction: int  # 0 or 1
    n_segments: int
    n_high_risk_segments: int
    segments: List[SegmentPrediction]
    top_risk_segments: List[SegmentPrediction]
    timeline: List[Dict]  # For visualization


class SegmentAnalyzer:
    """Analyze audio/video segments for suicide risk indicators"""
    
    def __init__(self, model_path: str = None):
        """Load segment model
        
        Args:
            model_path: Path to segment_model.pkl. If None, uses default location.
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, 'segment_model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Segment model not found: {model_path}")
        
        model_package = joblib.load(model_path)
        
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.feature_names = model_package['feature_names']
        self.audio_features = model_package['audio_features']
        self.text_features = model_package['text_features']
        self.threshold = model_package['threshold']
        self.model_type = model_package.get('model_type', 'Unknown')
        
        print(f"Loaded {self.model_type}")
        print(f"Features: {len(self.feature_names)} ({len(self.audio_features)} audio, {len(self.text_features)} text)")
        print(f"Threshold: {self.threshold:.4f}")
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to risk level"""
        if score < 0.3:
            return 'low'
        elif score < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _get_feature_contributions(self, X_row: np.ndarray) -> List[Dict]:
        """Get top features contributing to prediction for a single sample
        
        For tree models, we can use feature importance * feature value direction
        """
        try:
            # Get base feature importance from model
            if hasattr(self.model, 'get_feature_importance'):
                importance = self.model.get_feature_importance()
            elif hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            else:
                return []
            
            # Calculate contribution as importance * normalized value
            X_scaled = self.scaler.transform(X_row.reshape(1, -1))[0]
            contributions = importance * np.abs(X_scaled)
            
            # Get top contributing features
            top_indices = np.argsort(contributions)[-10:][::-1]
            
            result = []
            for idx in top_indices:
                feat_name = self.feature_names[idx]
                feat_type = 'audio' if feat_name in self.audio_features else 'text'
                result.append({
                    'feature': feat_name,
                    'type': feat_type,
                    'value': float(X_row[idx]),
                    'contribution': float(contributions[idx]),
                    'importance': float(importance[idx])
                })
            
            return result
            
        except Exception as e:
            print(f"Warning: Could not compute feature contributions: {e}")
            return []
    
    def predict_segments(self, df: pd.DataFrame) -> List[SegmentPrediction]:
        """Predict risk for each segment in dataframe
        
        Args:
            df: DataFrame with segment data (must have feature columns + metadata)
            
        Returns:
            List of SegmentPrediction objects
        """
        # Ensure all features are present
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing, filling with 0")
            for f in missing_features:
                df[f] = 0
        
        # Prepare features
        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Build predictions
        predictions = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Get segment metadata
            segment_id = row.get('segment_id', f'segment_{i}')
            start_time = row.get('start', row.get('start_time', 0))
            end_time = row.get('end', row.get('end_time', 0))
            duration = row.get('duration', end_time - start_time)
            text = row.get('text', '')
            
            risk_score = probabilities[i]
            risk_level = self._get_risk_level(risk_score)
            
            # Get feature contributions for this segment
            top_features = self._get_feature_contributions(X[i])
            
            predictions.append(SegmentPrediction(
                segment_id=str(segment_id),
                start_time=float(start_time),
                end_time=float(end_time),
                duration=float(duration),
                text=str(text) if pd.notna(text) else '',
                risk_score=float(risk_score),
                risk_level=risk_level,
                top_features=top_features
            ))
        
        return predictions
    
    def analyze_session(
        self, 
        df: pd.DataFrame, 
        session_id: str = None,
        aggregation: str = 'mean_top5'
    ) -> SessionAnalysis:
        """Analyze a complete session
        
        Args:
            df: DataFrame with segment data for one session
            session_id: Session identifier
            aggregation: How to aggregate segment scores ('mean', 'max', 'mean_top5')
            
        Returns:
            SessionAnalysis with full breakdown
        """
        if session_id is None:
            session_id = df.get('file_id', df.get('session_id', ['unknown'])).iloc[0]
        
        # Get segment predictions
        segments = self.predict_segments(df)
        
        # Sort by time
        segments = sorted(segments, key=lambda s: s.start_time)
        
        # Calculate session score
        scores = [s.risk_score for s in segments]
        
        if aggregation == 'mean':
            overall_score = np.mean(scores)
        elif aggregation == 'max':
            overall_score = np.max(scores)
        elif aggregation == 'mean_top5':
            top_k = min(5, len(scores))
            overall_score = np.mean(sorted(scores, reverse=True)[:top_k])
        else:
            overall_score = np.mean(scores)
        
        # Get high risk segments
        high_risk = [s for s in segments if s.risk_level == 'high']
        
        # Get top 5 risk segments
        top_risk = sorted(segments, key=lambda s: s.risk_score, reverse=True)[:5]
        
        # Build timeline for visualization
        timeline = []
        for s in segments:
            timeline.append({
                'start': s.start_time,
                'end': s.end_time,
                'score': s.risk_score,
                'level': s.risk_level,
                'text_preview': s.text[:100] if s.text else ''
            })
        
        return SessionAnalysis(
            session_id=str(session_id),
            overall_risk_score=float(overall_score),
            overall_risk_level=self._get_risk_level(overall_score),
            prediction=1 if overall_score >= self.threshold else 0,
            n_segments=len(segments),
            n_high_risk_segments=len(high_risk),
            segments=segments,
            top_risk_segments=top_risk,
            timeline=timeline
        )
    
    def format_report(self, analysis: SessionAnalysis) -> str:
        """Format analysis as human-readable report
        
        Args:
            analysis: SessionAnalysis object
            
        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"ANALYSIS REPORT: {analysis.session_id}")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall result
        result_text = "RISK DETECTED" if analysis.prediction == 1 else "LOW RISK"
        lines.append(f"OVERALL RESULT: {result_text}")
        lines.append(f"Risk Score: {analysis.overall_risk_score:.1%}")
        lines.append(f"Risk Level: {analysis.overall_risk_level.upper()}")
        lines.append("")
        
        # Summary
        lines.append(f"Total segments analyzed: {analysis.n_segments}")
        lines.append(f"High-risk segments: {analysis.n_high_risk_segments}")
        lines.append("")
        
        # Top risk segments
        lines.append("-" * 80)
        lines.append("TOP RISK SEGMENTS (triggers)")
        lines.append("-" * 80)
        
        for i, seg in enumerate(analysis.top_risk_segments, 1):
            start_fmt = self._format_time(seg.start_time)
            end_fmt = self._format_time(seg.end_time)
            
            lines.append(f"\n{i}. [{start_fmt} - {end_fmt}] Risk: {seg.risk_score:.1%} ({seg.risk_level.upper()})")
            
            if seg.text:
                text_preview = seg.text[:200]
                if len(seg.text) > 200:
                    text_preview += "..."
                lines.append(f"   Text: \"{text_preview}\"")
            
            if seg.top_features:
                lines.append("   Key indicators:")
                for feat in seg.top_features[:3]:
                    feat_name = feat['feature'].replace('_', ' ').replace('sma3nz', '').replace('amean', 'avg')
                    lines.append(f"     - {feat_name}: {feat['value']:.3f}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def to_dict(self, analysis: SessionAnalysis) -> Dict:
        """Convert analysis to dictionary for JSON serialization"""
        return {
            'session_id': analysis.session_id,
            'overall_risk_score': analysis.overall_risk_score,
            'overall_risk_level': analysis.overall_risk_level,
            'prediction': analysis.prediction,
            'prediction_label': 'RISK' if analysis.prediction == 1 else 'CONTROL',
            'n_segments': analysis.n_segments,
            'n_high_risk_segments': analysis.n_high_risk_segments,
            'threshold': self.threshold,
            'top_risk_segments': [
                {
                    'segment_id': s.segment_id,
                    'time_range': f"{self._format_time(s.start_time)} - {self._format_time(s.end_time)}",
                    'start_seconds': s.start_time,
                    'end_seconds': s.end_time,
                    'duration': s.duration,
                    'risk_score': s.risk_score,
                    'risk_level': s.risk_level,
                    'text': s.text,
                    'top_features': s.top_features[:5]
                }
                for s in analysis.top_risk_segments
            ],
            'timeline': analysis.timeline
        }


def analyze_from_csv(csv_path: str, model_path: str = None) -> SessionAnalysis:
    """Convenience function to analyze a CSV file
    
    Args:
        csv_path: Path to CSV with segment features
        model_path: Optional path to model
        
    Returns:
        SessionAnalysis
    """
    analyzer = SegmentAnalyzer(model_path)
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Get session ID from filename or data
    if 'file_id' in df.columns:
        session_id = df['file_id'].iloc[0]
    else:
        session_id = Path(csv_path).stem
    
    return analyzer.analyze_session(df, session_id)


def main():
    """Demo analysis on test data"""
    
    print("Segment Analyzer Demo")
    print("=" * 80)
    
    # Load segment model
    try:
        analyzer = SegmentAnalyzer()
    except FileNotFoundError:
        print("Segment model not found. Train it first with:")
        print("  python3 src/training/train_segment_model.py")
        return
    
    # Load test data
    merged_path = os.path.join(DATA_DIR, 'merged_features.csv')
    if not os.path.exists(merged_path):
        print(f"Test data not found: {merged_path}")
        return
    
    df = pd.read_csv(merged_path, low_memory=False)
    
    # Analyze first session
    first_session = df['file_id'].iloc[0]
    session_df = df[df['file_id'] == first_session].copy()
    
    print(f"\nAnalyzing session: {first_session}")
    print(f"Segments: {len(session_df)}")
    
    analysis = analyzer.analyze_session(session_df, first_session)
    
    # Print report
    report = analyzer.format_report(analysis)
    print(report)
    
    # Also show as dict
    print("\nJSON-compatible output:")
    result_dict = analyzer.to_dict(analysis)
    print(f"Session: {result_dict['session_id']}")
    print(f"Prediction: {result_dict['prediction_label']}")
    print(f"Risk Score: {result_dict['overall_risk_score']:.1%}")


if __name__ == '__main__':
    main()
