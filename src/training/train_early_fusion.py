#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
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

# Варианты обработки loudness-признаков:
# 'exclude' - удалить полностью
# 'normalize' - z-нормализация по всему датасету (убирает абсолютные значения)
# 'keep' - оставить как есть
LOUDNESS_MODE = 'keep'

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("warning: catboost not installed. will use only linearsvc")


def optimize_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold


def train_early_fusion_linear_svc(X_text, X_num, y, groups, cv_splits):
    print("\n" + "="*80)
    print("early fusion: tf-idf + linear svc")
    print("="*80)
    
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    oof_predictions = np.zeros(len(X_text))
    oof_labels = y.copy()
    
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
    
    print(f"tf-idf max_features: {tfidf_config['max_features']}")
    print(f"numeric features: {X_num.shape[1]}")
    print(f"total features: {tfidf_config['max_features']} + {X_num.shape[1]} = {tfidf_config['max_features'] + X_num.shape[1]}")
    
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
    
    print(f"\n[mean metrics early fusion (linear svc)]")
    print(f"  roc-auc: {mean_roc:.4f} | pr-auc: {mean_pr:.4f} | f1: {mean_f1:.4f}")
    
    print("\n[training final model on all data]")
    vectorizer_final = TfidfVectorizer(**tfidf_config)
    X_tfidf_final = vectorizer_final.fit_transform(X_text)
    
    scaler_final = StandardScaler()
    X_num_final_scaled = scaler_final.fit_transform(X_num)
    
    X_final_combined = hstack([X_tfidf_final, X_num_final_scaled])
    
    base_clf_final = LinearSVC(C=0.5, class_weight='balanced', max_iter=8000, random_state=RANDOM_STATE)
    clf_final = CalibratedClassifierCV(base_clf_final, method='sigmoid', cv=3)
    clf_final.fit(X_final_combined, y)
    
    print("model trained")
    
    final_model = {
        'vectorizer': vectorizer_final,
        'scaler': scaler_final,
        'classifier': clf_final,
        'model_type': 'LinearSVC'
    }
    
    return final_model, metrics, oof_predictions, oof_labels


def train_early_fusion_catboost(X_text, X_num, y, groups, cv_splits):
    print("\n" + "="*80)
    print("early fusion: catboost (numeric only)")
    print("="*80)
    
    print("note: catboost uses only numeric features (text_num + audio_num)")
    print(f"feature count: {X_num.shape[1]}")
    
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    metrics = []
    oof_predictions = np.zeros(len(X_num))
    oof_labels = y.copy()
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_num, y, groups), start=1):
        print(f"\n[fold {fold_idx}/{cv_splits}]")
        
        X_train, X_val = X_num[train_idx], X_num[val_idx]
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
    
    print(f"\n[mean metrics early fusion (catboost)]")
    print(f"  roc-auc: {mean_roc:.4f} | pr-auc: {mean_pr:.4f} | f1: {mean_f1:.4f}")
    
    print("\n[training final model on all data]")
    X_num_clean = np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    
    model_final.fit(X_num_clean, y)
    print("model trained")
    
    final_model = {
        'model': model_final,
        'model_type': 'CatBoost'
    }
    
    return final_model, metrics, oof_predictions, oof_labels


def main():
    print("\n" + "="*80)
    print("early fusion: multimodal model (audio + text)")
    print("="*80)
    
    data_path = os.path.join(DATA_DIR, 'multi_session_df.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"loaded dataset: {len(df)} sessions")
    
    audio_cols = [c for c in df.columns if c.startswith('audio_') and c not in ['audio_label']]
    text_num_cols = [c for c in df.columns if c.startswith('text_') and c not in ['text_label']]
    
    # Обработка loudness-признаков
    loudness_cols = [c for c in audio_cols if 'loudness' in c.lower()]
    
    if LOUDNESS_MODE == 'exclude':
        audio_cols = [c for c in audio_cols if c not in loudness_cols]
        print(f"excluded {len(loudness_cols)} loudness features")
    elif LOUDNESS_MODE == 'normalize':
        # Z-нормализация loudness-признаков (убирает влияние абсолютной громкости)
        for col in loudness_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val
        print(f"normalized {len(loudness_cols)} loudness features (z-score)")
    else:
        print(f"keeping {len(loudness_cols)} loudness features as-is")
    
    X_text = df['full_text'].fillna("").astype(str)
    X_audio = df[audio_cols].fillna(0).values
    X_text_num = df[text_num_cols].fillna(0).values
    
    X_num_combined = np.hstack([X_text_num, X_audio])
    
    y = df['label'].values
    groups = df['session_id'].values
    
    print(f"audio features: {len(audio_cols)}")
    print(f"numeric text features: {len(text_num_cols)}")
    print(f"total numeric features: {X_num_combined.shape[1]}")
    print(f"class balance: {(y == 0).sum()} control, {(y == 1).sum()} risk")
    
    # Используем 10 фолдов для более стабильной оценки на малых данных
    cv_splits = min(10, len(df) // 8)  # Минимум 8 сессий на фолд
    cv_splits = max(cv_splits, 3)  # Минимум 3 фолда
    print(f"\nusing {cv_splits}-fold groupkfold cv")
    
    linear_model, linear_metrics, linear_oof_pred, linear_oof_labels = train_early_fusion_linear_svc(
        X_text, X_num_combined, y, groups, cv_splits
    )
    
    if HAS_CATBOOST:
        catboost_model, catboost_metrics, catboost_oof_pred, catboost_oof_labels = train_early_fusion_catboost(
            X_text, X_num_combined, y, groups, cv_splits
        )
    else:
        catboost_model = None
        catboost_metrics = None
    
    print("\n" + "="*80)
    print("saving models")
    print("="*80)
    
    early_fusion_linear_path = os.path.join(MODELS_DIR, 'early_fusion_linear_svc.pkl')
    joblib.dump({
        'model': linear_model,
        'audio_feature_names': audio_cols,
        'text_feature_names': text_num_cols
    }, early_fusion_linear_path)
    print(f"linear svc model saved: {early_fusion_linear_path}")
    
    metadata = {
        'model_type': 'Early Fusion (Audio + Text)',
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_samples': int(len(df)),
        'n_audio_features': len(audio_cols),
        'n_text_features': len(text_num_cols),
        'n_total_numeric_features': X_num_combined.shape[1],
        'cv_folds': cv_splits,
        'linear_svc_metrics': {
            'mean_roc_auc': float(np.mean([m['roc_auc'] for m in linear_metrics])),
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in linear_metrics])),
            'mean_f1': float(np.mean([m['f1'] for m in linear_metrics])),
            'mean_threshold': float(np.mean([m['threshold'] for m in linear_metrics])),
            'fold_results': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v 
                 for k, v in m.items()}
                for m in linear_metrics
            ],
            'oof_predictions': linear_oof_pred.tolist(),
            'oof_labels': linear_oof_labels.tolist()
        },
        'random_state': RANDOM_STATE
    }
    
    if catboost_model is not None:
        early_fusion_catboost_path = os.path.join(MODELS_DIR, 'early_fusion_catboost.pkl')
        joblib.dump({
            'model': catboost_model,
            'audio_feature_names': audio_cols,
            'text_feature_names': text_num_cols
        }, early_fusion_catboost_path)
        print(f"catboost model saved: {early_fusion_catboost_path}")
        
        metadata['catboost_metrics'] = {
            'mean_roc_auc': float(np.mean([m['roc_auc'] for m in catboost_metrics])),
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in catboost_metrics])),
            'mean_f1': float(np.mean([m['f1'] for m in catboost_metrics])),
            'mean_threshold': float(np.mean([m['threshold'] for m in catboost_metrics])),
            'fold_results': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v 
                 for k, v in m.items()}
                for m in catboost_metrics
            ],
            'oof_predictions': catboost_oof_pred.tolist(),
            'oof_labels': catboost_oof_labels.tolist()
        }
    
    metadata_path = os.path.join(RESULTS_DIR, 'early_fusion_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"metadata saved: {metadata_path}")
    
    print("\n" + "="*80)
    print("early fusion: training completed")
    print("="*80)
    print(f"\nearly fusion metrics:")
    print(f"\n  [linear svc (tf-idf + numeric)]")
    print(f"    roc-auc: {metadata['linear_svc_metrics']['mean_roc_auc']:.4f}")
    print(f"    pr-auc:  {metadata['linear_svc_metrics']['mean_pr_auc']:.4f}")
    print(f"    f1:      {metadata['linear_svc_metrics']['mean_f1']:.4f}")
    print(f"    threshold: {metadata['linear_svc_metrics']['mean_threshold']:.3f}")
    
    if catboost_model is not None:
        print(f"\n  [catboost (numeric only)]")
        print(f"    roc-auc: {metadata['catboost_metrics']['mean_roc_auc']:.4f}")
        print(f"    pr-auc:  {metadata['catboost_metrics']['mean_pr_auc']:.4f}")
        print(f"    f1:      {metadata['catboost_metrics']['mean_f1']:.4f}")
        print(f"    threshold: {metadata['catboost_metrics']['mean_threshold']:.3f}")


if __name__ == '__main__':
    main()

