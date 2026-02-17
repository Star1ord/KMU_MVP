#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Late Fusion Model:
- Model 1: TF-IDF + LinearSVC (text semantics)
- Model 2: CatBoost (audio features)
- Meta-model: LogisticRegression on OOF predictions

This approach lets each modality learn its own patterns before combining.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)
import joblib
import json

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: CatBoost not installed")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def optimize_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def train_text_model_oof(X_text, y, groups, cv_splits):
    """Train TF-IDF + LinearSVC and return OOF predictions"""
    print("\n" + "="*80)
    print("Stage 1: Text model (TF-IDF + LinearSVC)")
    print("="*80)
    
    tfidf_config = {
        'sublinear_tf': True,
        'strip_accents': 'unicode',
        'analyzer': 'word',
        'token_pattern': r'\w{1,}',
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
        'max_features': 5000
    }
    
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(X_text))
    metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_text, y, groups), start=1):
        X_train = X_text.iloc[train_idx]
        X_val = X_text.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        vectorizer = TfidfVectorizer(**tfidf_config)
        X_tr = vectorizer.fit_transform(X_train)
        X_va = vectorizer.transform(X_val)
        
        base_clf = LinearSVC(C=0.5, class_weight='balanced', max_iter=8000, random_state=RANDOM_STATE)
        clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
        clf.fit(X_tr, y_train)
        
        y_proba = clf.predict_proba(X_va)[:, 1]
        oof_predictions[val_idx] = y_proba
        
        roc_auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        
        metrics.append({'fold': fold_idx, 'roc_auc': roc_auc, 'pr_auc': pr_auc})
        print(f"[fold {fold_idx}] roc-auc: {roc_auc:.4f} | pr-auc: {pr_auc:.4f}")
    
    mean_roc = np.mean([m['roc_auc'] for m in metrics])
    mean_pr = np.mean([m['pr_auc'] for m in metrics])
    print(f"\n[Text model mean] roc-auc: {mean_roc:.4f} | pr-auc: {mean_pr:.4f}")
    
    # Train final text model
    vectorizer_final = TfidfVectorizer(**tfidf_config)
    X_all = vectorizer_final.fit_transform(X_text)
    base_clf_final = LinearSVC(C=0.5, class_weight='balanced', max_iter=8000, random_state=RANDOM_STATE)
    clf_final = CalibratedClassifierCV(base_clf_final, method='sigmoid', cv=3)
    clf_final.fit(X_all, y)
    
    return oof_predictions, {'vectorizer': vectorizer_final, 'classifier': clf_final}, metrics


def train_audio_model_oof(X_audio, y, groups, cv_splits):
    """Train CatBoost on audio features and return OOF predictions"""
    print("\n" + "="*80)
    print("Stage 2: Audio model (CatBoost)")
    print("="*80)
    
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(X_audio))
    metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_audio, y, groups), start=1):
        X_train, X_val = X_audio[train_idx], X_audio[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        model = cb.CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=3,
            random_seed=RANDOM_STATE,
            verbose=False,
            loss_function='Logloss',
            eval_metric='AUC',
            colsample_bylevel=0.8,
            min_data_in_leaf=5,
            max_leaves=16,
            grow_policy='Lossguide',
            bootstrap_type='Bayesian',
            bagging_temperature=0.7,
            task_type='CPU',
            thread_count=-1
        )
        
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = y_proba
        
        roc_auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        
        metrics.append({'fold': fold_idx, 'roc_auc': roc_auc, 'pr_auc': pr_auc})
        print(f"[fold {fold_idx}] roc-auc: {roc_auc:.4f} | pr-auc: {pr_auc:.4f}")
    
    mean_roc = np.mean([m['roc_auc'] for m in metrics])
    mean_pr = np.mean([m['pr_auc'] for m in metrics])
    print(f"\n[Audio model mean] roc-auc: {mean_roc:.4f} | pr-auc: {mean_pr:.4f}")
    
    # Train final audio model
    X_audio_clean = np.nan_to_num(X_audio, nan=0.0, posinf=0.0, neginf=0.0)
    model_final = cb.CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        verbose=False,
        loss_function='Logloss',
        eval_metric='AUC',
        colsample_bylevel=0.8,
        min_data_in_leaf=5,
        max_leaves=16,
        grow_policy='Lossguide',
        bootstrap_type='Bayesian',
        bagging_temperature=0.7,
        task_type='CPU',
        thread_count=-1
    )
    model_final.fit(X_audio_clean, y)
    
    return oof_predictions, model_final, metrics


def train_meta_model(p_text_oof, p_audio_oof, y, groups, cv_splits):
    """Train meta-model on OOF predictions from both modalities"""
    print("\n" + "="*80)
    print("Stage 3: Meta-model (LogisticRegression)")
    print("="*80)
    
    X_meta = np.column_stack([p_text_oof, p_audio_oof])
    
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(X_meta))
    metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta, y, groups), start=1):
        X_train, X_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        meta_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='liblinear',
            random_state=RANDOM_STATE,
            max_iter=1000
        )
        meta_model.fit(X_train, y_train)
        
        y_proba = meta_model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = y_proba
        
        roc_auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        
        best_thr = optimize_threshold(y_val, y_proba)
        y_pred = (y_proba >= best_thr).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        # Get meta-model weights
        coefs = meta_model.coef_[0]
        
        metrics.append({
            'fold': fold_idx,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'threshold': best_thr,
            'coef_text': coefs[0],
            'coef_audio': coefs[1]
        })
        print(f"[fold {fold_idx}] roc-auc: {roc_auc:.4f} | pr-auc: {pr_auc:.4f} | f1: {f1:.4f}")
        print(f"           weights: text={coefs[0]:.3f}, audio={coefs[1]:.3f}")
    
    mean_roc = np.mean([m['roc_auc'] for m in metrics])
    mean_pr = np.mean([m['pr_auc'] for m in metrics])
    mean_f1 = np.mean([m['f1'] for m in metrics])
    mean_threshold = np.mean([m['threshold'] for m in metrics])
    
    # Average weights
    avg_text_weight = np.mean([m['coef_text'] for m in metrics])
    avg_audio_weight = np.mean([m['coef_audio'] for m in metrics])
    
    print(f"\n[Meta-model mean] roc-auc: {mean_roc:.4f} | pr-auc: {mean_pr:.4f} | f1: {mean_f1:.4f}")
    print(f"[Avg weights] text: {avg_text_weight:.3f} | audio: {avg_audio_weight:.3f}")
    
    # Train final meta-model
    meta_final = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        solver='liblinear',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    meta_final.fit(X_meta, y)
    
    return oof_predictions, meta_final, metrics, mean_threshold


def main():
    print("\n" + "="*80)
    print("LATE FUSION MODEL")
    print("Text (TF-IDF + SVC) + Audio (CatBoost) -> Meta (LogReg)")
    print("="*80)
    
    if not HAS_CATBOOST:
        raise ImportError("CatBoost is required for this model")
    
    # Load data
    data_path = os.path.join(DATA_DIR, 'multi_session_df.csv')
    df = pd.read_csv(data_path)
    print(f"\nLoaded: {len(df)} sessions")
    print(f"Balance: {(df['label'] == 0).sum()} control, {(df['label'] == 1).sum()} risk")
    
    # Prepare data
    X_text = df['full_text'].fillna("").astype(str)
    
    audio_cols = [c for c in df.columns if c.startswith('audio_')]
    X_audio = df[audio_cols].fillna(0).values
    
    y = df['label'].values
    groups = df['session_id'].values
    
    print(f"Text: avg {X_text.str.len().mean():.0f} chars")
    print(f"Audio: {X_audio.shape[1]} features")
    
    cv_splits = min(10, len(df) // 8)
    cv_splits = max(cv_splits, 3)
    print(f"\nUsing {cv_splits}-fold GroupKFold CV")
    
    # Stage 1: Text model
    p_text_oof, text_model, text_metrics = train_text_model_oof(X_text, y, groups, cv_splits)
    
    # Stage 2: Audio model
    p_audio_oof, audio_model, audio_metrics = train_audio_model_oof(X_audio, y, groups, cv_splits)
    
    # Stage 3: Meta-model
    p_meta_oof, meta_model, meta_metrics, mean_threshold = train_meta_model(
        p_text_oof, p_audio_oof, y, groups, cv_splits
    )
    
    # Confusion matrix
    print("\n" + "="*80)
    print("CONFUSION MATRIX (Late Fusion)")
    print("="*80)
    
    y_pred = (p_meta_oof >= mean_threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n                    Predicted")
    print(f"                 CONTROL  RISK")
    print(f"Actual CONTROL     {tn:3d}    {fp:3d}")
    print(f"Actual RISK        {fn:3d}    {tp:3d}")
    
    accuracy = (tp + tn) / len(y)
    oof_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    oof_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    oof_f1 = 2 * oof_precision * oof_recall / (oof_precision + oof_recall) if (oof_precision + oof_recall) > 0 else 0
    
    print(f"\nAccuracy:  {accuracy:.3f}")
    print(f"Precision: {oof_precision:.3f}")
    print(f"Recall:    {oof_recall:.3f}")
    print(f"F1-Score:  {oof_f1:.3f}")
    
    # Save models
    print("\n" + "="*80)
    print("Saving models")
    print("="*80)
    
    model_package = {
        'text_model': text_model,
        'audio_model': audio_model,
        'meta_model': meta_model,
        'audio_feature_names': audio_cols,
        'model_type': 'Late Fusion (Text + Audio)'
    }
    
    model_path = os.path.join(MODELS_DIR, 'late_fusion_v2.pkl')
    joblib.dump(model_package, model_path)
    print(f"Model saved: {model_path}")
    
    # Metadata
    metadata = {
        'model_type': 'Late Fusion (Text TF-IDF + Audio CatBoost)',
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_samples': int(len(df)),
        'n_audio_features': len(audio_cols),
        'cv_folds': cv_splits,
        'text_metrics': {
            'mean_roc_auc': float(np.mean([m['roc_auc'] for m in text_metrics])),
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in text_metrics]))
        },
        'audio_metrics': {
            'mean_roc_auc': float(np.mean([m['roc_auc'] for m in audio_metrics])),
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in audio_metrics]))
        },
        'meta_metrics': {
            'mean_roc_auc': float(np.mean([m['roc_auc'] for m in meta_metrics])),
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in meta_metrics])),
            'mean_f1': float(np.mean([m['f1'] for m in meta_metrics])),
            'mean_threshold': float(mean_threshold),
            'avg_text_weight': float(np.mean([m['coef_text'] for m in meta_metrics])),
            'avg_audio_weight': float(np.mean([m['coef_audio'] for m in meta_metrics])),
            'fold_results': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v 
                 for k, v in m.items()}
                for m in meta_metrics
            ],
            'oof_predictions': p_meta_oof.tolist(),
            'oof_labels': y.tolist()
        },
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'random_state': RANDOM_STATE
    }
    
    metadata_path = os.path.join(RESULTS_DIR, 'late_fusion_v2_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved: {metadata_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nText model (TF-IDF + SVC):   ROC-AUC {metadata['text_metrics']['mean_roc_auc']:.4f}")
    print(f"Audio model (CatBoost):      ROC-AUC {metadata['audio_metrics']['mean_roc_auc']:.4f}")
    print(f"Late Fusion (meta):          ROC-AUC {metadata['meta_metrics']['mean_roc_auc']:.4f}")
    print(f"\nFinal metrics:")
    print(f"  ROC-AUC: {metadata['meta_metrics']['mean_roc_auc']:.4f}")
    print(f"  PR-AUC:  {metadata['meta_metrics']['mean_pr_auc']:.4f}")
    print(f"  F1:      {metadata['meta_metrics']['mean_f1']:.4f}")
    print(f"\nModality weights: text={metadata['meta_metrics']['avg_text_weight']:.3f}, audio={metadata['meta_metrics']['avg_audio_weight']:.3f}")


if __name__ == '__main__':
    main()







