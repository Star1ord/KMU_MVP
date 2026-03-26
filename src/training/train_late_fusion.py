#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_curve, f1_score,
    precision_score, recall_score
)
from scipy.sparse import hstack
import joblib
import json
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("error: catboost not installed. install: pip install catboost")
    exit(1)


def optimize_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold


def train_audio_model_oof(X_audio, y, groups, cv_splits):
    print("\n" + "="*80)
    print("training audio model (catboost) with oof")
    print("="*80)
    
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(X_audio))
    models = []
    metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_audio, y, groups), start=1):
        print(f"\n[fold {fold_idx}/{cv_splits}]")
        
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
        
        best_thr = optimize_threshold(y_val, y_proba)
        y_pred = (y_proba >= best_thr).astype(int)
        
        f1 = f1_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        print(f"  roc-auc: {roc_auc:.4f} | pr-auc: {pr_auc:.4f} | f1: {f1:.4f}")
        print(f"  precision: {precision:.4f} | recall: {recall:.4f} | threshold: {best_thr:.3f}")
        
        models.append(model)
        metrics.append({
            'fold': fold_idx,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'threshold': best_thr
        })
    
    mean_roc = np.mean([m['roc_auc'] for m in metrics])
    mean_pr = np.mean([m['pr_auc'] for m in metrics])
    mean_f1 = np.mean([m['f1'] for m in metrics])
    
    print(f"\n[mean metrics audio]")
    print(f"  roc-auc: {mean_roc:.4f} | pr-auc: {mean_pr:.4f} | f1: {mean_f1:.4f}")
    
    return oof_predictions, models, metrics


def train_text_model_oof(X_text, X_num, y, groups, cv_splits):
    print("\n" + "="*80)
    print("training text model (tf-idf + linear svc) with oof")
    print("="*80)
    
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(X_text))
    models = []
    metrics = []
    
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
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_text, y, groups), start=1):
        print(f"\n[fold {fold_idx}/{cv_splits}]")
        
        X_text_train, X_text_val = X_text.iloc[train_idx], X_text.iloc[val_idx]
        X_num_train, X_num_val = X_num[train_idx], X_num[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        vectorizer = TfidfVectorizer(**tfidf_config)
        X_tfidf_train = vectorizer.fit_transform(X_text_train)
        X_tfidf_val = vectorizer.transform(X_text_val)
        
        scaler = StandardScaler()
        X_num_train_scaled = scaler.fit_transform(X_num_train)
        X_num_val_scaled = scaler.transform(X_num_val)
        
        X_train_combined = hstack([X_tfidf_train, X_num_train_scaled])
        X_val_combined = hstack([X_tfidf_val, X_num_val_scaled])
        
        base_clf = LinearSVC(C=0.5, class_weight='balanced', max_iter=8000, random_state=RANDOM_STATE)
        clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
        clf.fit(X_train_combined, y_train)
        
        y_proba = clf.predict_proba(X_val_combined)[:, 1]
        oof_predictions[val_idx] = y_proba
        
        roc_auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        
        best_thr = optimize_threshold(y_val, y_proba)
        y_pred = (y_proba >= best_thr).astype(int)
        
        f1 = f1_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        print(f"  roc-auc: {roc_auc:.4f} | pr-auc: {pr_auc:.4f} | f1: {f1:.4f}")
        print(f"  precision: {precision:.4f} | recall: {recall:.4f} | threshold: {best_thr:.3f}")
        
        models.append({
            'vectorizer': vectorizer,
            'scaler': scaler,
            'classifier': clf
        })
        metrics.append({
            'fold': fold_idx,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'threshold': best_thr
        })
    
    mean_roc = np.mean([m['roc_auc'] for m in metrics])
    mean_pr = np.mean([m['pr_auc'] for m in metrics])
    mean_f1 = np.mean([m['f1'] for m in metrics])
    
    print(f"\n[mean metrics text]")
    print(f"  roc-auc: {mean_roc:.4f} | pr-auc: {mean_pr:.4f} | f1: {mean_f1:.4f}")
    
    return oof_predictions, models, metrics


def train_meta_model(p_audio_oof, p_text_oof, y, groups, cv_splits):
    print("\n" + "="*80)
    print("training meta model (logistic regression)")
    print("="*80)
    
    X_meta = np.column_stack([p_audio_oof, p_text_oof])
    
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_meta, y, groups), start=1):
        print(f"\n[fold {fold_idx}/{cv_splits}]")
        
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
        
        roc_auc = roc_auc_score(y_val, y_proba)
        pr_auc = average_precision_score(y_val, y_proba)
        
        best_thr = optimize_threshold(y_val, y_proba)
        y_pred = (y_proba >= best_thr).astype(int)
        
        f1 = f1_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        print(f"  roc-auc: {roc_auc:.4f} | pr-auc: {pr_auc:.4f} | f1: {f1:.4f}")
        print(f"  precision: {precision:.4f} | recall: {recall:.4f} | threshold: {best_thr:.3f}")
        
        coefs = meta_model.coef_[0]
        print(f"  meta model weights: audio={coefs[0]:.3f}, text={coefs[1]:.3f}")
        
        metrics.append({
            'fold': fold_idx,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'threshold': best_thr,
            'coef_audio': coefs[0],
            'coef_text': coefs[1]
        })
    
    mean_roc = np.mean([m['roc_auc'] for m in metrics])
    mean_pr = np.mean([m['pr_auc'] for m in metrics])
    mean_f1 = np.mean([m['f1'] for m in metrics])
    
    print(f"\n[mean metrics meta model]")
    print(f"  roc-auc: {mean_roc:.4f} | pr-auc: {mean_pr:.4f} | f1: {mean_f1:.4f}")
    
    final_meta_model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        solver='liblinear',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    final_meta_model.fit(X_meta, y)
    
    print(f"\n[final meta model]")
    final_coefs = final_meta_model.coef_[0]
    print(f"  weights: audio={final_coefs[0]:.3f}, text={final_coefs[1]:.3f}")
    
    return final_meta_model, metrics


def train_final_models(X_audio, X_text, X_num, y):
    print("\n" + "="*80)
    print("training final models on all data")
    print("="*80)
    
    print("\n[audio model]")
    X_audio_clean = np.nan_to_num(X_audio, nan=0.0, posinf=0.0, neginf=0.0)
    
    final_audio_model = cb.CatBoostClassifier(
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
    final_audio_model.fit(X_audio_clean, y)
    print("audio model trained")
    
    print("\n[text model]")
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
    
    vectorizer = TfidfVectorizer(**tfidf_config)
    X_tfidf = vectorizer.fit_transform(X_text)
    
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    X_combined = hstack([X_tfidf, X_num_scaled])
    
    base_clf = LinearSVC(C=0.5, class_weight='balanced', max_iter=8000, random_state=RANDOM_STATE)
    clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    clf.fit(X_combined, y)
    
    print("text model trained")
    
    final_text_pipeline = {
        'vectorizer': vectorizer,
        'scaler': scaler,
        'classifier': clf
    }
    
    return final_audio_model, final_text_pipeline


def main():
    print("\n" + "="*80)
    print("late fusion: multimodal model (audio + text)")
    print("="*80)
    
    data_path = os.path.join(DATA_DIR, 'multi_session_df.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"loaded dataset: {len(df)} sessions")
    
    audio_cols = [c for c in df.columns if c.startswith('audio_') and c not in ['audio_label']]
    text_num_cols = [c for c in df.columns if c.startswith('text_') and c not in ['text_label']]
    
    X_audio = df[audio_cols].fillna(0).values
    X_text = df['full_text'].fillna("").astype(str)
    X_num = df[text_num_cols].fillna(0).values
    y = df['label'].values
    groups = df['session_id'].values
    
    print(f"audio features: {len(audio_cols)}")
    print(f"numeric text features: {len(text_num_cols)}")
    print(f"class balance: {(y == 0).sum()} control, {(y == 1).sum()} risk")
    
    cv_splits = 5
    print(f"\nusing {cv_splits}-fold groupkfold cv")
    
    p_audio_oof, audio_models, audio_metrics = train_audio_model_oof(
        X_audio, y, groups, cv_splits
    )
    
    p_text_oof, text_models, text_metrics = train_text_model_oof(
        X_text, X_num, y, groups, cv_splits
    )
    
    meta_model, meta_metrics = train_meta_model(
        p_audio_oof, p_text_oof, y, groups, cv_splits
    )
    
    final_audio_model, final_text_pipeline = train_final_models(
        X_audio, X_text, X_num, y
    )
    
    print("\n" + "="*80)
    print("saving models")
    print("="*80)
    
    late_fusion_package = {
        'audio_model': final_audio_model,
        'text_pipeline': final_text_pipeline,
        'meta_model': meta_model,
        'audio_feature_names': audio_cols,
        'text_feature_names': text_num_cols,
        'tfidf_config': final_text_pipeline['vectorizer'].get_params()
    }
    
    model_path = os.path.join(MODELS_DIR, 'late_fusion_model.pkl')
    joblib.dump(late_fusion_package, model_path)
    print(f"model saved: {model_path}")
    
    metadata = {
        'model_type': 'Late Fusion (Audio + Text)',
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_samples': int(len(df)),
        'n_audio_features': len(audio_cols),
        'n_text_features': len(text_num_cols),
        'cv_folds': cv_splits,
        'audio_metrics': {
            'mean_roc_auc': float(np.mean([m['roc_auc'] for m in audio_metrics])),
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in audio_metrics])),
            'mean_recall': float(np.mean([m['recall'] for m in audio_metrics])),
            'mean_f1': float(np.mean([m['f1'] for m in audio_metrics])),
            'fold_results': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v 
                 for k, v in m.items()}
                for m in audio_metrics
            ]
        },
        'text_metrics': {
            'mean_roc_auc': float(np.mean([m['roc_auc'] for m in text_metrics])),
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in text_metrics])),
            'mean_recall': float(np.mean([m['recall'] for m in text_metrics])),
            'mean_f1': float(np.mean([m['f1'] for m in text_metrics])),
            'fold_results': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v 
                 for k, v in m.items()}
                for m in text_metrics
            ]
        },
        'meta_metrics': {
            'mean_roc_auc': float(np.mean([m['roc_auc'] for m in meta_metrics])),
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in meta_metrics])),
            'mean_recall': float(np.mean([m['recall'] for m in meta_metrics])),
            'mean_f1': float(np.mean([m['f1'] for m in meta_metrics])),
            'mean_threshold': float(np.mean([m['threshold'] for m in meta_metrics])),
            'fold_results': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v 
                 for k, v in m.items()}
                for m in meta_metrics
            ]
        },
        'random_state': RANDOM_STATE
    }
    
    metadata_path = os.path.join(RESULTS_DIR, 'late_fusion_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"metadata saved: {metadata_path}")
    
    print("\n" + "="*80)
    print("late fusion: training completed")
    print("="*80)
    print(f"\nmodel comparison:")
    print(f"  [audio]")
    print(f"    roc-auc: {metadata['audio_metrics']['mean_roc_auc']:.4f}")
    print(f"    pr-auc:  {metadata['audio_metrics']['mean_pr_auc']:.4f}")
    print(f"    f1:      {metadata['audio_metrics']['mean_f1']:.4f}")
    print(f"\n  [text]")
    print(f"    roc-auc: {metadata['text_metrics']['mean_roc_auc']:.4f}")
    print(f"    pr-auc:  {metadata['text_metrics']['mean_pr_auc']:.4f}")
    print(f"    f1:      {metadata['text_metrics']['mean_f1']:.4f}")
    print(f"\n  [meta model (fusion)]")
    print(f"    roc-auc: {metadata['meta_metrics']['mean_roc_auc']:.4f}")
    print(f"    pr-auc:  {metadata['meta_metrics']['mean_pr_auc']:.4f}")
    print(f"    f1:      {metadata['meta_metrics']['mean_f1']:.4f}")
    print(f"    threshold: {metadata['meta_metrics']['mean_threshold']:.3f}")


if __name__ == '__main__':
    main()
