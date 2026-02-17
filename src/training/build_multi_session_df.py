#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import DATA_DIR

warnings.filterwarnings('ignore')


def extract_base_file_id(file_id):
    return re.sub(r'\s*\(\d+\)$', '', str(file_id))


def build_audio_session_df(audio_path):
    print("\n" + "="*80)
    print("aggregating audio features by sessions")
    print("="*80)
    
    df = pd.read_csv(audio_path)
    print(f"loaded {len(df)} audio segments")
    
    df['session_id'] = df['file_id'].apply(extract_base_file_id)
    df = df.sort_values(['session_id', 'start']).reset_index(drop=True)
    
    exclude_cols = [
        'file_id', 'segment_id', 'start', 'end', 'duration',
        'asr_conf_mean', 'asr_conf_std', 'word_count', 'segment_path',
        'label', 'session_id'
    ]
    
    audio_feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"audio features count: {len(audio_feature_cols)}")
    
    aggregated_data = []
    
    for session_id in df['session_id'].unique():
        session_data = df[df['session_id'] == session_id].copy()
        label = session_data['label'].iloc[0]
        n_segments = len(session_data)
        
        agg_features = {
            'session_id': session_id,
            'label': label,
            'n_audio_segments': n_segments,
            'total_audio_duration': session_data['duration'].sum()
        }
        
        for feat in audio_feature_cols:
            values = session_data[feat].values
            values = values[~np.isnan(values)]
            
            if len(values) == 0:
                agg_features[f'audio_{feat}_mean'] = 0.0
                agg_features[f'audio_{feat}_std'] = 0.0
                agg_features[f'audio_{feat}_median'] = 0.0
                agg_features[f'audio_{feat}_min'] = 0.0
                agg_features[f'audio_{feat}_max'] = 0.0
            else:
                agg_features[f'audio_{feat}_mean'] = np.mean(values)
                agg_features[f'audio_{feat}_std'] = np.std(values) if len(values) > 1 else 0.0
                agg_features[f'audio_{feat}_median'] = np.median(values)
                agg_features[f'audio_{feat}_min'] = np.min(values)
                agg_features[f'audio_{feat}_max'] = np.max(values)
        
        aggregated_data.append(agg_features)
    
    audio_session_df = pd.DataFrame(aggregated_data)
    
    print(f"aggregated sessions: {len(audio_session_df)}")
    print(f"features per session: {len(audio_session_df.columns) - 2}")
    print(f"class balance: {(audio_session_df['label'] == 0).sum()} control, "
          f"{(audio_session_df['label'] == 1).sum()} risk")
    
    return audio_session_df


def build_text_session_df(text_path):
    print("\n" + "="*80)
    print("aggregating text features by sessions")
    print("="*80)
    
    df = pd.read_csv(text_path)
    print(f"loaded {len(df)} text segments")
    
    df = df.sort_values(['session_id', 'start_time']).reset_index(drop=True)
    
    exclude_cols = {
        'session_id', 'label', 'speaker', 'text',
        'start_time', 'end_time'
    }
    
    num_cols = [
        c for c in df.columns 
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    
    print(f"numeric text features count: {len(num_cols)}")
    
    agg_funcs = ['mean', 'std', 'min', 'max', 'median']
    
    text_num_agg = df.groupby('session_id')[num_cols].agg(agg_funcs)
    text_num_agg.columns = [f'text_{c}_{stat}' for c, stat in text_num_agg.columns]
    text_num_agg = text_num_agg.reset_index()
    
    text_concat = df.groupby('session_id')['text'].apply(
        lambda texts: ' '.join(
            str(t).strip() 
            for t in texts 
            if isinstance(t, str) and pd.notna(t) and str(t).strip()
        )
    ).reset_index()
    text_concat.columns = ['session_id', 'full_text']
    
    label_agg = df.groupby('session_id')['label'].first().reset_index()
    
    n_segments = df.groupby('session_id').size().reset_index(name='n_text_segments')
    
    text_session_df = text_num_agg.merge(text_concat, on='session_id', how='left')
    text_session_df = text_session_df.merge(label_agg, on='session_id', how='left')
    text_session_df = text_session_df.merge(n_segments, on='session_id', how='left')
    
    print(f"aggregated sessions: {len(text_session_df)}")
    print(f"features per session: {len(text_session_df.columns) - 2}")
    print(f"class balance: {(text_session_df['label'] == 0).sum()} control, "
          f"{(text_session_df['label'] == 1).sum()} risk")
    
    return text_session_df


def merge_session_dfs(audio_df, text_df):
    print("\n" + "="*80)
    print("merging audio and text features")
    print("="*80)
    
    audio_sessions = set(audio_df['session_id'])
    text_sessions = set(text_df['session_id'])
    
    common_sessions = audio_sessions & text_sessions
    audio_only = audio_sessions - text_sessions
    text_only = text_sessions - audio_sessions
    
    print(f"sessions only in audio: {len(audio_only)}")
    print(f"sessions only in text: {len(text_only)}")
    print(f"common sessions: {len(common_sessions)}")
    
    overlap_pct = len(common_sessions) / max(len(audio_sessions), len(text_sessions)) * 100
    print(f"overlap percentage: {overlap_pct:.1f}%")
    
    if overlap_pct < 50:
        print("warning: session overlap less than 50%")
    
    multi_df = audio_df.merge(
        text_df,
        on='session_id',
        how='inner',
        suffixes=('_audio_label', '_text_label')
    )
    
    if 'label_audio_label' in multi_df.columns and 'label_text_label' in multi_df.columns:
        label_mismatch = (multi_df['label_audio_label'] != multi_df['label_text_label']).sum()
        
        if label_mismatch > 0:
            print(f"error: found {label_mismatch} label mismatches")
            mismatch_sessions = multi_df[
                multi_df['label_audio_label'] != multi_df['label_text_label']
            ]['session_id'].tolist()
            print(f"error: sessions with mismatches: {mismatch_sessions[:5]}...")
            raise ValueError("label mismatch detected between audio and text data")
        
        multi_df['label'] = multi_df['label_audio_label']
        multi_df = multi_df.drop(['label_audio_label', 'label_text_label'], axis=1)
    
    nan_summary = multi_df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]
    
    if len(nan_cols) > 0:
        print(f"warning: found nan in {len(nan_cols)} features")
        print(f"top-5 features with nan:")
        print(nan_cols.nlargest(5))
    
    print(f"\nfinal dataset:")
    print(f"  - sessions: {len(multi_df)}")
    print(f"  - total features: {len(multi_df.columns)}")
    print(f"  - class balance: {(multi_df['label'] == 0).sum()} control, "
          f"{(multi_df['label'] == 1).sum()} risk")
    
    audio_cols = [c for c in multi_df.columns if c.startswith('audio_')]
    text_cols = [c for c in multi_df.columns if c.startswith('text_')]
    
    print(f"  - audio features: {len(audio_cols)}")
    print(f"  - numeric text features: {len(text_cols)}")
    print(f"  - text (full_text): {'yes' if 'full_text' in multi_df.columns else 'no'}")
    
    return multi_df


def validate_dataset(multi_df):
    print("\n" + "="*80)
    print("dataset validation")
    print("="*80)
    
    checks_passed = True
    
    if 'session_id' not in multi_df.columns:
        print("error: missing session_id column")
        checks_passed = False
    else:
        duplicates = multi_df['session_id'].duplicated().sum()
        if duplicates > 0:
            print(f"error: found duplicate session_ids: {duplicates}")
            checks_passed = False
        else:
            print("ok: no duplicate session_ids")
    
    if 'label' not in multi_df.columns:
        print("error: missing label column")
        checks_passed = False
    else:
        unique_labels = multi_df['label'].unique()
        if not np.array_equal(sorted(unique_labels), [0, 1]):
            print(f"error: invalid labels: {unique_labels}")
            checks_passed = False
        else:
            print("ok: labels are valid (0, 1)")
    
    if 'full_text' not in multi_df.columns:
        print("warning: missing full_text column")
    else:
        empty_text = multi_df['full_text'].isna() | (multi_df['full_text'] == '')
        if empty_text.sum() > 0:
            print(f"warning: empty text in {empty_text.sum()} sessions")
        else:
            print("ok: all sessions have text")
    
    audio_cols = [c for c in multi_df.columns if c.startswith('audio_')]
    text_cols = [c for c in multi_df.columns if c.startswith('text_')]
    
    if len(audio_cols) == 0:
        print("error: no audio features")
        checks_passed = False
    else:
        print(f"ok: audio features present: {len(audio_cols)}")
    
    if len(text_cols) == 0:
        print("warning: no numeric text features")
    else:
        print(f"ok: numeric text features present: {len(text_cols)}")
    
    min_sessions = 30
    if len(multi_df) < min_sessions:
        print(f"warning: too few sessions for reliable training: {len(multi_df)} < {min_sessions}")
    else:
        print(f"ok: enough sessions for training: {len(multi_df)}")
    
    min_class_samples = 10
    class_counts = multi_df['label'].value_counts()
    for label, count in class_counts.items():
        if count < min_class_samples:
            print(f"warning: too few samples for class {label}: {count} < {min_class_samples}")
    
    print("\n" + "="*80)
    if checks_passed:
        print("validation passed")
    else:
        print("validation failed")
    print("="*80)
    
    return checks_passed


def main():
    print("\n" + "="*80)
    print("building multimodal session-level dataset")
    print("="*80)
    
    audio_path = os.path.join(DATA_DIR, 'opensmile_features.csv')
    text_path = os.path.join(DATA_DIR, 'text_features_nlp2_ready.csv')
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"file not found: {audio_path}")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"file not found: {text_path}")
    
    audio_session_df = build_audio_session_df(audio_path)
    
    text_session_df = build_text_session_df(text_path)
    
    multi_session_df = merge_session_dfs(audio_session_df, text_session_df)
    
    if not validate_dataset(multi_session_df):
        print("\nwarning: dataset has issues, but continuing with save...")
    
    output_path = os.path.join(DATA_DIR, 'multi_session_df.csv')
    multi_session_df.to_csv(output_path, index=False)
    print(f"dataset saved: {output_path}")
    
    output_audio_path = os.path.join(DATA_DIR, 'audio_session_df.csv')
    output_text_path = os.path.join(DATA_DIR, 'text_session_df.csv')
    audio_session_df.to_csv(output_audio_path, index=False)
    text_session_df.to_csv(output_text_path, index=False)
    print(f"intermediate datasets saved:")
    print(f"  - {output_audio_path}")
    print(f"  - {output_text_path}")
    
    print("\n" + "="*80)
    print("dataset building completed")
    print("="*80)
    print(f"final statistics:")
    print(f"  file: {output_path}")
    print(f"  sessions: {len(multi_session_df)}")
    print(f"  features: {len(multi_session_df.columns) - 2}")
    print(f"  class 0: {(multi_session_df['label'] == 0).sum()}")
    print(f"  class 1: {(multi_session_df['label'] == 1).sum()}")


if __name__ == '__main__':
    main()

