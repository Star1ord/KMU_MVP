#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment-Level Model for Suicide Risk Detection

Architecture:
1. Each segment gets its own prediction (audio + text features)
2. Segment scores are aggregated to session-level
3. Provides interpretability - which segments contributed most to the decision

Key Features:
- Segment-level predictions with timestamps
- Audio (eGeMAPS) + Text (NLP) fusion per segment
- Interpretable output: top risk segments identified
- GroupKFold CV to prevent data leakage
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)
import joblib
import json
import warnings

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: CatBoost not installed, using GradientBoostingClassifier")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_segment_data():
    """Load segment-level data with proper cleaning"""
    print("\n" + "="*80)
    print("Loading segment-level data")
    print("="*80)
    
    # Try merged_features first (has both audio and text)
    merged_path = os.path.join(DATA_DIR, 'merged_features.csv')
    
    if os.path.exists(merged_path):
        df = pd.read_csv(merged_path, low_memory=False)
        print(f"Loaded merged_features.csv: {len(df)} segments")
    else:
        # Fallback: load and merge separate files
        audio_path = os.path.join(DATA_DIR, 'opensmile_features.csv')
        text_path = os.path.join(DATA_DIR, 'text_features_nlp2_ready.csv')
        
        audio_df = pd.read_csv(audio_path)
        text_df = pd.read_csv(text_path)
        
        # Merge on session and segment
        df = audio_df.merge(
            text_df,
            left_on=['file_id', 'segment_id'],
            right_on=['session_id', 'segment_id'],
            how='inner'
        )
        print(f"Merged audio ({len(audio_df)}) and text ({len(text_df)}) -> {len(df)} segments")
    
    # Clean problematic columns
    cols_to_drop = []
    for col in df.columns:
        # Remove duplicate labels
        if col in ['label_x', 'label_y']:
            cols_to_drop.append(col)
        # Remove path columns
        if 'path' in col.lower():
            cols_to_drop.append(col)
    
    # Keep one label column
    if 'label' not in df.columns:
        if 'label_x' in df.columns:
            df['label'] = df['label_x']
        elif 'label_y' in df.columns:
            df['label'] = df['label_y']
    
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Ensure file_id exists
    if 'file_id' not in df.columns and 'session_id' in df.columns:
        df['file_id'] = df['session_id']
    
    print(f"Sessions: {df['file_id'].nunique()}")
    print(f"Segments: {len(df)}")
    print(f"Balance: {(df['label'] == 0).sum()} control, {(df['label'] == 1).sum()} risk segments")
    
    return df


def get_feature_columns(df):
    """Extract audio and text feature column names"""
    
    # Columns to exclude from features (not useful for prediction)
    exclude_cols = {
        'file_id', 'session_id', 'segment_id', 'label', 'label_x', 'label_y',
        'text', 'start', 'end', 'speaker',
        'audio_path', 'segment_path', 'transcript_path',
        'asr_conf_mean', 'asr_conf_std'  # ASR confidence not useful
    }
    
    all_cols = df.columns.tolist()
    
    # eGeMAPS audio feature patterns
    egemaps_patterns = [
        'F0', 'loudness', 'mfcc', 'jitter', 'shimmer', 
        'HNR', 'logRel', 'frequency', 'bandwidth', 'amplitude',
        'alpha', 'hammarberg', 'slope', 'spectral', 'Voiced', 'Unvoiced',
        'equivalent', 'Peaks'
    ]
    
    # Text/NLP feature names (exact match or pattern)
    text_feature_names = {
        'avg_word_length', 'articulation_rate', 'negation_count',
        'pause_ratio', 'pronoun_count', 'speech_rate', 
        'type_token_ratio', 'word_count', 'duration'
    }
    
    audio_features = []
    text_features = []
    
    for col in all_cols:
        # Skip excluded columns
        if col in exclude_cols:
            continue
        
        # Check if it's an eGeMAPS feature
        is_egemaps = any(pat in col for pat in egemaps_patterns)
        
        # Check if it's a text feature
        is_text = col in text_feature_names
        
        if is_egemaps:
            audio_features.append(col)
        elif is_text:
            text_features.append(col)
    
    print(f"\nFeatures identified:")
    print(f"  Audio (eGeMAPS): {len(audio_features)}")
    print(f"  Text (NLP): {len(text_features)}")
    if text_features:
        print(f"  Text features: {text_features}")
    
    return audio_features, text_features


def train_segment_model(df, audio_features, text_features, cv_splits=10):
    """Train segment-level model with cross-validation"""
    
    print("\n" + "="*80)
    print("Training Segment-Level Model")
    print("="*80)
    
    # Prepare features (audio_features already cleaned in main)
    all_features = audio_features + text_features
    X = df[all_features].copy()
    y = df['label'].values
    groups = df['file_id'].values
    
    # Fill NaN with 0 and handle inf
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Store segment info for later
    segment_info = df[['file_id', 'segment_id', 'start', 'end', 'text']].copy() \
        if 'text' in df.columns else df[['file_id', 'segment_id', 'start', 'end']].copy()
    
    print(f"Training data: {len(X)} segments, {len(all_features)} features")
    print(f"Groups (sessions): {len(np.unique(groups))}")
    
    # Check class balance at session level
    session_labels = df.groupby('file_id')['label'].first()
    print(f"Session-level balance: {(session_labels == 0).sum()} control, {(session_labels == 1).sum()} risk")
    
    # Cross-validation
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    oof_predictions = np.zeros(len(X))
    fold_metrics = []
    feature_importance = np.zeros(len(all_features))
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model with better hyperparameters
        if HAS_CATBOOST:
            model = cb.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=5,
                loss_function='Logloss',
                auto_class_weights='Balanced',  # Balance classes
                random_seed=RANDOM_STATE,
                verbose=False,
                early_stopping_rounds=50
            )
            model.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                verbose=False
            )
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            feature_importance += model.get_feature_importance() / cv_splits
        else:
            model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=RANDOM_STATE
            )
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            feature_importance += model.feature_importances_ / cv_splits
        
        oof_predictions[val_idx] = y_proba
        
        # Metrics
        roc_auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        
        fold_metrics.append({
            'fold': fold_idx,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        })
        
        print(f"[Fold {fold_idx}] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    
    # Overall metrics
    mean_roc = np.mean([m['roc_auc'] for m in fold_metrics])
    mean_pr = np.mean([m['pr_auc'] for m in fold_metrics])
    
    # Optimal threshold based on balanced accuracy
    from sklearn.metrics import balanced_accuracy_score
    thresholds = np.linspace(0.1, 0.9, 100)
    balanced_accs = []
    for thresh in thresholds:
        y_pred = (oof_predictions >= thresh).astype(int)
        ba = balanced_accuracy_score(y, y_pred)
        balanced_accs.append(ba)
    
    optimal_threshold = thresholds[np.argmax(balanced_accs)]
    
    print(f"\n[Mean] ROC-AUC: {mean_roc:.4f} | PR-AUC: {mean_pr:.4f}")
    print(f"Optimal threshold (balanced accuracy): {optimal_threshold:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': feature_importance,
        'type': ['audio' if f in audio_features else 'text' for f in all_features]
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 most important features:")
    for _, row in importance_df.head(15).iterrows():
        print(f"  [{row['type']}] {row['feature']}: {row['importance']:.4f}")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    if HAS_CATBOOST:
        final_model = cb.CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=5,
            loss_function='Logloss',
            auto_class_weights='Balanced',
            random_seed=RANDOM_STATE,
            verbose=False
        )
    else:
        final_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE
        )
    
    final_model.fit(X_scaled, y)
    
    return {
        'model': final_model,
        'scaler': scaler_final,
        'feature_names': all_features,
        'audio_features': audio_features,
        'text_features': text_features,
        'threshold': optimal_threshold,
        'feature_importance': importance_df,
        'oof_predictions': oof_predictions,
        'fold_metrics': fold_metrics,
        'mean_roc_auc': mean_roc,
        'mean_pr_auc': mean_pr
    }


def aggregate_to_session(df, segment_predictions, method='percentile_90'):
    """Aggregate segment predictions to session level
    
    Methods:
    - mean: Average of all segments
    - max: Maximum segment score
    - mean_top5: Average of top 5 highest scoring segments
    - percentile_90: 90th percentile of segment scores (more robust)
    - weighted: Weighted by segment duration
    """
    
    df = df.copy()
    df['segment_score'] = segment_predictions
    
    session_results = []
    
    for session_id in df['file_id'].unique():
        session_data = df[df['file_id'] == session_id].copy()
        session_data = session_data.sort_values('start')
        
        scores = session_data['segment_score'].values
        true_label = session_data['label'].iloc[0]
        
        if method == 'mean':
            session_score = np.mean(scores)
        elif method == 'max':
            session_score = np.max(scores)
        elif method == 'mean_top5':
            top_k = min(5, len(scores))
            session_score = np.mean(np.sort(scores)[-top_k:])
        elif method == 'percentile_90':
            session_score = np.percentile(scores, 90)
        elif method == 'weighted':
            durations = session_data['duration'].values if 'duration' in session_data.columns else np.ones(len(scores))
            session_score = np.average(scores, weights=durations)
        else:
            session_score = np.mean(scores)
        
        # Get top risk segments
        top_segments = session_data.nlargest(5, 'segment_score')[[
            'segment_id', 'start', 'end', 'segment_score'
        ]].to_dict('records')
        
        session_results.append({
            'session_id': session_id,
            'session_score': session_score,
            'true_label': true_label,
            'n_segments': len(scores),
            'max_segment_score': np.max(scores),
            'min_segment_score': np.min(scores),
            'std_segment_score': np.std(scores),
            'top_risk_segments': top_segments
        })
    
    return pd.DataFrame(session_results)


def evaluate_session_predictions(session_df, threshold=0.5):
    """Evaluate session-level predictions"""
    
    y_true = session_df['true_label'].values
    y_proba = session_df['session_score'].values
    y_pred = (y_proba >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*80)
    print("Session-Level Evaluation")
    print("="*80)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              CONTROL  RISK")
    print(f"Actual CONTROL  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"Actual RISK     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist()
    }


def analyze_feature_quality(df, audio_features, text_features):
    """Analyze feature quality and potential issues"""
    
    print("\n" + "="*80)
    print("Feature Quality Analysis")
    print("="*80)
    
    all_features = audio_features + text_features
    
    # Check for features with very high correlation with label (suspicious)
    print("\nChecking for suspicious correlations with label...")
    suspicious = []
    
    for feat in all_features:
        if feat in df.columns:
            corr = df[[feat, 'label']].corr().iloc[0, 1]
            if abs(corr) > 0.5:
                suspicious.append((feat, corr))
    
    if suspicious:
        print("WARNING: Features with suspiciously high correlation (>0.5):")
        for feat, corr in sorted(suspicious, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {feat}: {corr:.4f}")
    else:
        print("OK: No suspiciously high correlations found")
    
    # Check for constant or near-constant features
    print("\nChecking for constant features...")
    constant_features = []
    for feat in all_features:
        if feat in df.columns:
            std = df[feat].std()
            if std < 1e-10:
                constant_features.append(feat)
    
    if constant_features:
        print(f"WARNING: {len(constant_features)} constant features found")
    else:
        print("OK: No constant features")
    
    # Check loudness variation (common normalization issue)
    print("\nLoudness analysis (checking for recording quality bias)...")
    loudness_cols = [c for c in audio_features if 'loudness' in c.lower()]
    
    for col in loudness_cols[:2]:
        if col in df.columns:
            by_label = df.groupby('label')[col].agg(['mean', 'std'])
            diff = abs(by_label.loc[0, 'mean'] - by_label.loc[1, 'mean'])
            print(f"  {col}:")
            print(f"    Control: {by_label.loc[0, 'mean']:.3f} +/- {by_label.loc[0, 'std']:.3f}")
            print(f"    Risk: {by_label.loc[1, 'mean']:.3f} +/- {by_label.loc[1, 'std']:.3f}")
            print(f"    Difference: {diff:.3f}")
            
            if diff > by_label['std'].mean() * 2:
                print("    WARNING: Large difference - possible recording quality bias!")
    
    return suspicious, constant_features


def main():
    print("\n" + "="*80)
    print("SEGMENT-LEVEL MODEL TRAINING")
    print("="*80)
    
    # Load data
    df = load_segment_data()
    
    # Get feature columns
    audio_features, text_features = get_feature_columns(df)
    
    if len(audio_features) == 0:
        print("ERROR: No audio features found!")
        return
    
    # REMOVE loudness features (unreliable across different recording quality)
    print("\n" + "="*80)
    print("REMOVING loudness features")
    print("="*80)
    loudness_cols = [f for f in audio_features if 'loudness' in f.lower() or 'equivalent' in f.lower()]
    print(f"Removing {len(loudness_cols)} loudness-related features")
    
    # Remove loudness features from audio_features list
    audio_features_clean = [f for f in audio_features if f not in loudness_cols]
    print(f"Audio features: {len(audio_features)} → {len(audio_features_clean)} (removed {len(loudness_cols)})")
    
    # Analyze feature quality
    analyze_feature_quality(df, audio_features_clean, text_features)
    
    # Train segment model WITHOUT loudness
    model_result = train_segment_model(df, audio_features_clean, text_features, cv_splits=10)
    
    # Aggregate to session level - try multiple methods
    print("\n" + "="*80)
    print("Aggregating to Session Level")
    print("="*80)
    
    # Try different aggregation methods
    methods = ['percentile_90', 'mean_top5', 'mean', 'max']
    best_method = None
    best_auc = 0
    
    for method in methods:
        session_df_temp = aggregate_to_session(df, model_result['oof_predictions'], method=method)
        y_true = session_df_temp['true_label'].values
        y_proba = session_df_temp['session_score'].values
        auc = roc_auc_score(y_true, y_proba)
        print(f"Method '{method}': Session ROC-AUC = {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_method = method
    
    print(f"\nBest aggregation method: {best_method} (AUC={best_auc:.4f})")
    session_df = aggregate_to_session(df, model_result['oof_predictions'], method=best_method)
    
    # Evaluate session predictions
    session_metrics = evaluate_session_predictions(session_df, model_result['threshold'])
    
    # Save model
    print("\n" + "="*80)
    print("Saving Model")
    print("="*80)
    
    model_package = {
        'model': model_result['model'],
        'scaler': model_result['scaler'],
        'feature_names': model_result['feature_names'],
        'audio_features': model_result['audio_features'],
        'text_features': model_result['text_features'],
        'threshold': model_result['threshold'],
        'model_type': 'Segment-Level CatBoost' if HAS_CATBOOST else 'Segment-Level GradientBoosting'
    }
    
    model_path = os.path.join(MODELS_DIR, 'segment_model.pkl')
    joblib.dump(model_package, model_path)
    print(f"Model saved: {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': model_package['model_type'],
        'n_segments': len(df),
        'n_sessions': df['file_id'].nunique(),
        'n_features': len(model_result['feature_names']),
        'n_audio_features': len(model_result['audio_features']),
        'n_text_features': len(model_result['text_features']),
        'segment_metrics': {
            'mean_roc_auc': model_result['mean_roc_auc'],
            'mean_pr_auc': model_result['mean_pr_auc'],
            'fold_results': model_result['fold_metrics']
        },
        'session_metrics': session_metrics,
        'threshold': model_result['threshold'],
        'top_features': model_result['feature_importance'].head(30).to_dict('records')
    }
    
    metadata_path = os.path.join(RESULTS_DIR, 'segment_model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved: {metadata_path}")
    
    # Save feature importance
    importance_path = os.path.join(RESULTS_DIR, 'segment_feature_importance.csv')
    model_result['feature_importance'].to_csv(importance_path, index=False)
    print(f"Feature importance saved: {importance_path}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Segment-level ROC-AUC: {model_result['mean_roc_auc']:.4f}")
    print(f"Session-level ROC-AUC: {session_metrics['roc_auc']:.4f}")
    print(f"Session-level F1: {session_metrics['f1']:.4f}")
    print(f"Threshold: {model_result['threshold']:.4f}")


if __name__ == '__main__':
    main()
