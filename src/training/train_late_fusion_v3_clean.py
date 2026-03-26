#!/usr/bin/env python3
"""
Late Fusion V3 - CLEAN VERSION
Без loudness признаков, с регуляризацией и правильной валидацией
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR

import joblib

RANDOM_STATE = 42
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
import catboost as cb
import json

print("\n" + "="*80)
print("LATE FUSION V3 - CLEAN VERSION")
print("Text (TF-IDF + SVC) + Audio (CatBoost) -> Meta (LogReg)")
print("БЕЗ LOUDNESS, С РЕГУЛЯРИЗАЦИЕЙ")
print("="*80)

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'multi_session_df.csv'))

print(f"\nLoaded: {len(df)} sessions")
print(f"Balance: {(df['label']==0).sum()} control, {(df['label']==1).sum()} risk")

# УДАЛЯЕМ ВСЕ LOUDNESS ПРИЗНАКИ
audio_cols = [c for c in df.columns if c.startswith('audio_')]
audio_cols = [c for c in audio_cols if 'loudness' not in c.lower() 
              and 'equivalent' not in c.lower() 
              and 'speech_rate' not in c
              and 'articulation' not in c
              and 'type_token' not in c
              and 'pronoun' not in c
              and 'negation' not in c]

print(f"\nAudio features (БЕЗ loudness): {len(audio_cols)}")
print(f"Text: avg {df['full_text'].str.len().mean():.0f} chars")

# Prepare data
full_text = df['full_text'].fillna('').astype(str)
X_audio = df[audio_cols].fillna(0).values
X_audio = np.nan_to_num(X_audio, nan=0.0, posinf=0.0, neginf=0.0)
y = df['label'].values
groups = np.arange(len(df))  # Каждая сессия - отдельная группа

# Cross-validation с 10 folds
cv_splits = 10
gkf = GroupKFold(n_splits=cv_splits)

print(f"\nUsing {cv_splits}-fold GroupKFold CV")

# ============================================================================
# Stage 1: Text Model (TF-IDF + LinearSVC)
# ============================================================================
print("\n" + "="*80)
print("Stage 1: Text model (TF-IDF + LinearSVC)")
print("="*80)

tfidf_config = {
    'sublinear_tf': True,
    'strip_accents': 'unicode',
    'analyzer': 'word',
    'token_pattern': r'\w{1,}',
    'ngram_range': (1, 2),
    'min_df': 3,  # Увеличили для борьбы с переобучением
    'max_df': 0.9,  # Уменьшили
    'max_features': 3000  # Уменьшили с 5000
}

oof_text = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(gkf.split(full_text, y, groups), 1):
    vectorizer = TfidfVectorizer(**tfidf_config, random_state=RANDOM_STATE)
    X_train_tfidf = vectorizer.fit_transform(full_text.iloc[train_idx])
    X_val_tfidf = vectorizer.transform(full_text.iloc[val_idx])
    
    # LinearSVC с большей регуляризацией
    base_clf = LinearSVC(
        C=0.3,  # Уменьшили с 0.5
        class_weight='balanced',
        max_iter=10000,
        random_state=RANDOM_STATE,
        dual=True  # Для малых датасетов
    )
    clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    clf.fit(X_train_tfidf, y[train_idx])
    
    oof_text[val_idx] = clf.predict_proba(X_val_tfidf)[:, 1]
    
    roc = roc_auc_score(y[val_idx], oof_text[val_idx])
    pr = average_precision_score(y[val_idx], oof_text[val_idx])
    print(f"[fold {fold}] roc-auc: {roc:.4f} | pr-auc: {pr:.4f}")

mean_roc_text = roc_auc_score(y, oof_text)
mean_pr_text = average_precision_score(y, oof_text)
print(f"\n[Text model mean] roc-auc: {mean_roc_text:.4f} | pr-auc: {mean_pr_text:.4f}")

# Train final text model
vectorizer_final = TfidfVectorizer(**tfidf_config, random_state=RANDOM_STATE)
X_tfidf_final = vectorizer_final.fit_transform(full_text)
base_clf_final = LinearSVC(C=0.3, class_weight='balanced', max_iter=10000, random_state=RANDOM_STATE, dual=True)
clf_text_final = CalibratedClassifierCV(base_clf_final, method='sigmoid', cv=3)
clf_text_final.fit(X_tfidf_final, y)

# ============================================================================
# Stage 2: Audio Model (CatBoost)
# ============================================================================
print("\n" + "="*80)
print("Stage 2: Audio model (CatBoost)")
print("="*80)

oof_audio = np.zeros(len(df))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_audio, y, groups), 1):
    model = cb.CatBoostClassifier(
        iterations=200,  # Уменьшили с 300
        depth=3,  # Уменьшили с 4
        learning_rate=0.05,  # Уменьшили с 0.03
        l2_leaf_reg=5,  # Увеличили регуляризацию
        random_seed=RANDOM_STATE,
        verbose=False,
        loss_function='Logloss',
        eval_metric='AUC',
        colsample_bylevel=0.7,  # Уменьшили с 0.8
        min_data_in_leaf=7,  # Увеличили с 5
        max_leaves=12,  # Уменьшили с 16
        grow_policy='Lossguide',
        bootstrap_type='Bayesian',
        bagging_temperature=0.5,  # Уменьшили с 0.7
        task_type='CPU',
        thread_count=-1
    )
    
    model.fit(X_audio[train_idx], y[train_idx])
    oof_audio[val_idx] = model.predict_proba(X_audio[val_idx])[:, 1]
    
    roc = roc_auc_score(y[val_idx], oof_audio[val_idx])
    pr = average_precision_score(y[val_idx], oof_audio[val_idx])
    print(f"[fold {fold}] roc-auc: {roc:.4f} | pr-auc: {pr:.4f}")

mean_roc_audio = roc_auc_score(y, oof_audio)
mean_pr_audio = average_precision_score(y, oof_audio)
print(f"\n[Audio model mean] roc-auc: {mean_roc_audio:.4f} | pr-auc: {mean_pr_audio:.4f}")

# Train final audio model
audio_model_final = cb.CatBoostClassifier(
    iterations=200,
    depth=3,
    learning_rate=0.05,
    l2_leaf_reg=5,
    random_seed=RANDOM_STATE,
    verbose=False,
    loss_function='Logloss',
    eval_metric='AUC',
    colsample_bylevel=0.7,
    min_data_in_leaf=7,
    max_leaves=12,
    grow_policy='Lossguide',
    bootstrap_type='Bayesian',
    bagging_temperature=0.5,
    task_type='CPU',
    thread_count=-1
)
audio_model_final.fit(X_audio, y)

# ============================================================================
# Stage 3: Meta-Model (LogisticRegression)
# ============================================================================
print("\n" + "="*80)
print("Stage 3: Meta-model (LogisticRegression)")
print("="*80)

X_meta = np.column_stack([oof_text, oof_audio])
oof_meta = np.zeros(len(df))
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_meta, y, groups), 1):
    meta_model = LogisticRegression(
        C=0.5,  # Уменьшили регуляризацию
        class_weight='balanced',
        solver='liblinear',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    
    meta_model.fit(X_meta[train_idx], y[train_idx])
    oof_meta[val_idx] = meta_model.predict_proba(X_meta[val_idx])[:, 1]
    
    roc = roc_auc_score(y[val_idx], oof_meta[val_idx])
    pr = average_precision_score(y[val_idx], oof_meta[val_idx])
    
    # Optimize threshold on F1
    thresholds = np.linspace(0.3, 0.7, 50)
    f1_scores = []
    for thr in thresholds:
        y_pred = (oof_meta[val_idx] >= thr).astype(int)
        f1_scores.append(f1_score(y[val_idx], y_pred))
    best_f1 = max(f1_scores)
    best_recall = recall_score(
        y[val_idx],
        (oof_meta[val_idx] >= thresholds[np.argmax(f1_scores)]).astype(int),
        zero_division=0,
    )
    
    # Get weights
    weights_text = meta_model.coef_[0][0]
    weights_audio = meta_model.coef_[0][1]
    
    print(f"[fold {fold}] roc-auc: {roc:.4f} | pr-auc: {pr:.4f} | f1: {best_f1:.4f}")
    print(f"           weights: text={weights_text:.3f}, audio={weights_audio:.3f}")
    
    fold_metrics.append({
        'roc_auc': roc,
        'pr_auc': pr,
        'recall': best_recall,
        'f1': best_f1
    })

mean_roc_meta = roc_auc_score(y, oof_meta)
mean_pr_meta = average_precision_score(y, oof_meta)
mean_recall_meta = np.mean([m['recall'] for m in fold_metrics])
mean_f1_meta = np.mean([m['f1'] for m in fold_metrics])

print(f"\n[Meta-model mean] roc-auc: {mean_roc_meta:.4f} | pr-auc: {mean_pr_meta:.4f} | f1: {mean_f1_meta:.4f}")

# Train final meta-model
meta_model_final = LogisticRegression(
    C=0.5,
    class_weight='balanced',
    solver='liblinear',
    random_state=RANDOM_STATE,
    max_iter=1000
)
meta_model_final.fit(X_meta, y)

# Optimize global threshold
thresholds = np.linspace(0.3, 0.7, 100)
f1_scores = []
for thr in thresholds:
    y_pred = (oof_meta >= thr).astype(int)
    f1_scores.append(f1_score(y, y_pred))

optimal_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"\nOptimal threshold: {optimal_threshold:.4f} (F1={best_f1:.4f})")

# ============================================================================
# Confusion Matrix
# ============================================================================
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

y_pred_final = (oof_meta >= optimal_threshold).astype(int)
cm = confusion_matrix(y, y_pred_final)

print("\n" + "="*80)
print("CONFUSION MATRIX (Late Fusion V3)")
print("="*80)
print(f"\n                    Predicted")
print(f"                 CONTROL  RISK")
print(f"Actual CONTROL      {cm[0,0]:3d}     {cm[0,1]:3d}")
print(f"Actual RISK         {cm[1,0]:3d}     {cm[1,1]:3d}")

print(f"\nAccuracy:  {accuracy_score(y, y_pred_final):.3f}")
print(f"Precision: {precision_score(y, y_pred_final):.3f}")
print(f"Recall:    {recall_score(y, y_pred_final):.3f}")
print(f"F1-Score:  {f1_score(y, y_pred_final):.3f}")

# ============================================================================
# Save Model
# ============================================================================
print("\n" + "="*80)
print("Saving models")
print("="*80)

model_package = {
    'text_model': {
        'vectorizer': vectorizer_final,
        'classifier': clf_text_final
    },
    'audio_model': audio_model_final,
    'meta_model': meta_model_final,
    'audio_feature_names': audio_cols,
    'threshold': optimal_threshold,
    'oof_predictions': {
        'text': oof_text,
        'audio': oof_audio,
        'meta': oof_meta
    },
    'fold_metrics': fold_metrics,
    'mean_metrics': {
        'text_roc_auc': mean_roc_text,
        'text_pr_auc': mean_pr_text,
        'audio_roc_auc': mean_roc_audio,
        'audio_pr_auc': mean_pr_audio,
        'meta_roc_auc': mean_roc_meta,
        'meta_pr_auc': mean_pr_meta,
        'meta_recall': mean_recall_meta,
        'meta_f1': mean_f1_meta
    }
}

model_path = os.path.join(MODELS_DIR, 'late_fusion_v3_clean.pkl')
joblib.dump(model_package, model_path)
print(f"Model saved: {model_path}")

# Save metadata
metadata = {
    'model_type': 'Late Fusion V3 Clean',
    'timestamp': pd.Timestamp.now().isoformat(),
    'n_samples': len(df),
    'n_audio_features': len(audio_cols),
    'loudness_removed': True,
    'cv_folds': cv_splits,
    'threshold': optimal_threshold,
    'metrics': {
        'text': {'roc_auc': mean_roc_text, 'pr_auc': mean_pr_text},
        'audio': {'roc_auc': mean_roc_audio, 'pr_auc': mean_pr_audio},
        'meta': {'roc_auc': mean_roc_meta, 'pr_auc': mean_pr_meta, 'recall': mean_recall_meta, 'f1': mean_f1_meta}
    }
}

metadata_path = os.path.join(RESULTS_DIR, 'late_fusion_v3_clean_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved: {metadata_path}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nText model (TF-IDF + SVC):   ROC-AUC {mean_roc_text:.4f}")
print(f"Audio model (CatBoost):      ROC-AUC {mean_roc_audio:.4f}")
print(f"Late Fusion V3 (meta):       ROC-AUC {mean_roc_meta:.4f}")
print(f"\nFinal metrics:")
print(f"  ROC-AUC: {mean_roc_meta:.4f}")
print(f"  PR-AUC:  {mean_pr_meta:.4f}")
print(f"  F1:      {mean_f1_meta:.4f}")
print(f"  Threshold: {optimal_threshold:.4f}")
print("\n✅ CLEAN MODEL WITHOUT LOUDNESS")
print("="*80)
