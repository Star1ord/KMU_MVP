#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Конвертер формата из my_pipeline в формат для ML-моделей.

Преобразует merged_features.csv (из pipeline) в:
- opensmile_features.csv (для аудио-модели)
- text_features_nlp2_ready.csv (для текстовой модели)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import DATA_DIR


def extract_opensmile_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекает OpenSMILE eGeMAPS признаки из merged_features.csv
    
    Args:
        merged_df: DataFrame из merged_features.csv
        
    Returns:
        DataFrame в формате opensmile_features.csv
    """
    opensmile_cols = [
        'file_id', 'segment_id', 'start', 'end', 'duration', 'label',
        'F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
        'F0semitoneFrom27.5Hz_sma3nz_percentile20.0', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
        'F0semitoneFrom27.5Hz_sma3nz_percentile80.0', 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
        'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
        'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope', 'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
        'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0',
        'loudness_sma3_percentile50.0', 'loudness_sma3_percentile80.0', 'loudness_sma3_pctlrange0-2',
        'loudness_sma3_meanRisingSlope', 'loudness_sma3_stddevRisingSlope',
        'loudness_sma3_meanFallingSlope', 'loudness_sma3_stddevFallingSlope',
        'spectralFlux_sma3_amean', 'spectralFlux_sma3_stddevNorm',
        'mfcc1_sma3_amean', 'mfcc1_sma3_stddevNorm', 'mfcc2_sma3_amean', 'mfcc2_sma3_stddevNorm',
        'mfcc3_sma3_amean', 'mfcc3_sma3_stddevNorm', 'mfcc4_sma3_amean', 'mfcc4_sma3_stddevNorm',
        'jitterLocal_sma3nz_amean', 'jitterLocal_sma3nz_stddevNorm',
        'shimmerLocaldB_sma3nz_amean', 'shimmerLocaldB_sma3nz_stddevNorm',
        'HNRdBACF_sma3nz_amean', 'HNRdBACF_sma3nz_stddevNorm',
        'logRelF0-H1-H2_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_stddevNorm',
        'logRelF0-H1-A3_sma3nz_amean', 'logRelF0-H1-A3_sma3nz_stddevNorm',
        'F1frequency_sma3nz_amean', 'F1frequency_sma3nz_stddevNorm',
        'F1bandwidth_sma3nz_amean', 'F1bandwidth_sma3nz_stddevNorm',
        'F1amplitudeLogRelF0_sma3nz_amean', 'F1amplitudeLogRelF0_sma3nz_stddevNorm',
        'F2frequency_sma3nz_amean', 'F2frequency_sma3nz_stddevNorm',
        'F2bandwidth_sma3nz_amean', 'F2bandwidth_sma3nz_stddevNorm',
        'F2amplitudeLogRelF0_sma3nz_amean', 'F2amplitudeLogRelF0_sma3nz_stddevNorm',
        'F3frequency_sma3nz_amean', 'F3frequency_sma3nz_stddevNorm',
        'F3bandwidth_sma3nz_amean', 'F3bandwidth_sma3nz_stddevNorm',
        'F3amplitudeLogRelF0_sma3nz_amean', 'F3amplitudeLogRelF0_sma3nz_stddevNorm',
        'alphaRatioV_sma3nz_amean', 'alphaRatioV_sma3nz_stddevNorm',
        'hammarbergIndexV_sma3nz_amean', 'hammarbergIndexV_sma3nz_stddevNorm',
        'slopeV0-500_sma3nz_amean', 'slopeV0-500_sma3nz_stddevNorm',
        'slopeV500-1500_sma3nz_amean', 'slopeV500-1500_sma3nz_stddevNorm',
        'spectralFluxV_sma3nz_amean', 'spectralFluxV_sma3nz_stddevNorm',
        'mfcc1V_sma3nz_amean', 'mfcc1V_sma3nz_stddevNorm',
        'mfcc2V_sma3nz_amean', 'mfcc2V_sma3nz_stddevNorm',
        'mfcc3V_sma3nz_amean', 'mfcc3V_sma3nz_stddevNorm',
        'mfcc4V_sma3nz_amean', 'mfcc4V_sma3nz_stddevNorm',
        'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean',
        'slopeUV0-500_sma3nz_amean', 'slopeUV500-1500_sma3nz_amean',
        'spectralFluxUV_sma3nz_amean', 'loudnessPeaksPerSec',
        'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec',
        'StddevVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength',
        'StddevUnvoicedSegmentLength', 'equivalentSoundLevel_dBp'
    ]
    
    available_cols = [col for col in opensmile_cols if col in merged_df.columns]
    missing_cols = [col for col in opensmile_cols if col not in merged_df.columns]
    
    if missing_cols:
        print(f"⚠️ Предупреждение: Отсутствуют {len(missing_cols)} eGeMAPS колонок")
    
    opensmile_df = merged_df[available_cols].copy()
    
    # Добавляем текстовые признаки, которые были вычислены в merge_features
    # Эти признаки нужны для правильной агрегации в audio_session_df
    text_feature_cols = [
        'speech_rate', 'articulation_rate', 'type_token_ratio',
        'pronoun_count', 'negation_count', 'avg_word_length'
    ]
    
    for col in text_feature_cols:
        if col in merged_df.columns and col not in opensmile_df.columns:
            opensmile_df[col] = merged_df[col]
    
    # Добавляем необходимые колонки для совместимости
    if 'segment_path' not in opensmile_df.columns:
        opensmile_df['segment_path'] = ''
    if 'asr_conf_mean' not in opensmile_df.columns:
        opensmile_df['asr_conf_mean'] = 0.0
    if 'asr_conf_std' not in opensmile_df.columns:
        opensmile_df['asr_conf_std'] = 0.0
    if 'word_count' not in opensmile_df.columns:
        opensmile_df['word_count'] = merged_df.get('text', '').apply(lambda x: len(str(x).split()))
    
    return opensmile_df


def extract_text_features(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекает текстовые NLP признаки из merged_features.csv
    
    Args:
        merged_df: DataFrame из merged_features.csv
        
    Returns:
        DataFrame в формате text_features_nlp2_ready.csv
    """
    text_df = merged_df[[
        'start', 'end', 'text', 'file_id', 'label'
    ]].copy()
    
    text_df.rename(columns={
        'start': 'start_time',
        'end': 'end_time',
        'file_id': 'session_id'
    }, inplace=True)
    
    text_df['duration'] = merged_df['duration']
    
    # Вычисляем pause_prev
    text_df = text_df.sort_values(['session_id', 'start_time']).reset_index(drop=True)
    text_df['pause_prev'] = text_df.groupby('session_id')['start_time'].diff()
    text_df.loc[text_df.groupby('session_id').head(1).index, 'pause_prev'] = np.nan
    
    # Добавляем текстовые признаки из merged_df
    text_features_mapping = {
        'speech_rate': 'words_per_sec',
        'articulation_rate': 'articulation_rate',
        'type_token_ratio': 'type_token_ratio',
        'pronoun_count': 'pronoun_count',
        'negation_count': 'negation_count',
        'avg_word_length': 'avg_word_len'
    }
    
    for merged_col, text_col in text_features_mapping.items():
        if merged_col in merged_df.columns:
            text_df[text_col] = merged_df[merged_col]
        else:
            text_df[text_col] = 0.0
    
    # Добавляем дополнительные признаки
    text_df['num_tokens'] = merged_df.get('word_count', text_df['text'].apply(lambda x: len(str(x).split())))
    text_df['num_unique_tokens'] = text_df['text'].apply(lambda x: len(set(str(x).lower().split())))
    text_df['num_chars'] = text_df['text'].apply(lambda x: len(str(x)))
    text_df['word_count'] = text_df['num_tokens']
    text_df['sent_count'] = text_df['text'].apply(lambda x: str(x).count('.') + str(x).count('!') + str(x).count('?') + 1)
    
    # Стопслова и сложные слова
    ru_stopwords = {
        'в', 'и', 'на', 'с', 'по', 'для', 'к', 'от', 'за', 'о', 'из', 'у', 'про',
        'это', 'как', 'но', 'что', 'так', 'он', 'она', 'они', 'мы', 'вы', 'я', 'ты'
    }
    
    def count_stopwords(text):
        words = str(text).lower().split()
        return sum(1 for w in words if w in ru_stopwords)
    
    def count_complex_words(text):
        words = str(text).split()
        return sum(1 for w in words if len(w) > 6)
    
    text_df['num_stopwords'] = text_df['text'].apply(count_stopwords)
    text_df['stopword_ratio'] = text_df['num_stopwords'] / (text_df['num_tokens'] + 1e-10)
    
    text_df['num_complex_words'] = text_df['text'].apply(count_complex_words)
    text_df['complex_word_ratio'] = text_df['num_complex_words'] / (text_df['num_tokens'] + 1e-10)
    
    # Знаки препинания
    text_df['qmark_count'] = text_df['text'].apply(lambda x: str(x).count('?'))
    text_df['excl_count'] = text_df['text'].apply(lambda x: str(x).count('!'))
    
    # Местоимения (детализированно)
    def count_i_pronoun(text):
        words = str(text).lower().split()
        i_pronouns = {'я', 'меня', 'мне', 'мной', 'мой', 'моя', 'моё', 'мои'}
        return sum(1 for w in words if w in i_pronouns)
    
    def count_we_pronoun(text):
        words = str(text).lower().split()
        we_pronouns = {'мы', 'нас', 'нам', 'нами', 'наш', 'наша', 'наше', 'наши'}
        return sum(1 for w in words if w in we_pronouns)
    
    def count_you_pronoun(text):
        words = str(text).lower().split()
        you_pronouns = {'ты', 'тебя', 'тебе', 'тобой', 'твой', 'твоя', 'твоё', 'твои', 'вы', 'вас', 'вам', 'вами', 'ваш', 'ваша', 'ваше', 'ваши'}
        return sum(1 for w in words if w in you_pronouns)
    
    text_df['i_pronoun'] = text_df['text'].apply(count_i_pronoun)
    text_df['we_pronoun'] = text_df['text'].apply(count_we_pronoun)
    text_df['you_pronoun'] = text_df['text'].apply(count_you_pronoun)
    
    # Speaker (неизвестно из pipeline)
    text_df['speaker'] = np.nan
    
    # Упорядочиваем колонки как в text_features_nlp2_ready.csv
    column_order = [
        'start_time', 'end_time', 'speaker', 'text', 'session_id', 'label',
        'duration', 'pause_prev', 'num_tokens', 'num_unique_tokens',
        'num_chars', 'word_count', 'sent_count', 'words_per_sec',
        'type_token_ratio', 'avg_word_len', 'num_complex_words',
        'complex_word_ratio', 'num_stopwords', 'stopword_ratio',
        'qmark_count', 'excl_count', 'i_pronoun', 'we_pronoun', 'you_pronoun'
    ]
    
    text_df = text_df[[col for col in column_order if col in text_df.columns]]
    
    return text_df


def convert_pipeline_format(
    merged_features_path: str,
    output_dir: str,
    overwrite: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Конвертирует merged_features.csv в форматы для моделей.
    
    Args:
        merged_features_path: Путь к merged_features.csv
        output_dir: Директория для сохранения результатов
        overwrite: Перезаписать существующие файлы или дополнить
        
    Returns:
        Tuple (opensmile_df, text_df)
    """
    print("\n" + "="*80)
    print("КОНВЕРТАЦИЯ ФОРМАТА PIPELINE → ML")
    print("="*80)
    
    if not os.path.exists(merged_features_path):
        raise FileNotFoundError(f"Файл не найден: {merged_features_path}")
    
    print(f"\n[1] Загрузка данных: {merged_features_path}")
    merged_df = pd.read_csv(merged_features_path)
    print(f"    Загружено: {len(merged_df)} сегментов")
    print(f"    Уникальных сессий: {merged_df['file_id'].nunique()}")
    
    print(f"\n[2] Извлечение OpenSMILE признаков...")
    opensmile_df = extract_opensmile_features(merged_df)
    print(f"    Признаков: {len(opensmile_df.columns)}")
    
    print(f"\n[3] Извлечение текстовых признаков...")
    text_df = extract_text_features(merged_df)
    print(f"    Признаков: {len(text_df.columns)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    opensmile_output = os.path.join(output_dir, 'opensmile_features.csv')
    text_output = os.path.join(output_dir, 'text_features_nlp2_ready.csv')
    
    if not overwrite:
        # Дополняем существующие файлы
        if os.path.exists(opensmile_output):
            print(f"\n[4] Объединение с существующим opensmile_features.csv...")
            existing_opensmile = pd.read_csv(opensmile_output)
            new_sessions = set(opensmile_df['file_id'].unique())
            existing_opensmile = existing_opensmile[~existing_opensmile['file_id'].isin(new_sessions)]
            opensmile_df = pd.concat([existing_opensmile, opensmile_df], ignore_index=True)
            print(f"    Итого сессий: {opensmile_df['file_id'].nunique()}")
        
        if os.path.exists(text_output):
            print(f"\n[5] Объединение с существующим text_features_nlp2_ready.csv...")
            existing_text = pd.read_csv(text_output)
            new_sessions = set(text_df['session_id'].unique())
            existing_text = existing_text[~existing_text['session_id'].isin(new_sessions)]
            text_df = pd.concat([existing_text, text_df], ignore_index=True)
            print(f"    Итого сессий: {text_df['session_id'].nunique()}")
    
    print(f"\n[6] Сохранение результатов...")
    opensmile_df.to_csv(opensmile_output, index=False)
    print(f"    ✅ {opensmile_output}")
    
    text_df.to_csv(text_output, index=False)
    print(f"    ✅ {text_output}")
    
    # Сохраняем merged_features в model_data_dir для будущих объединений
    merged_output = os.path.join(output_dir, 'merged_features.csv')
    merged_df.to_csv(merged_output, index=False)
    print(f"    ✅ {merged_output} (сохранен для будущих объединений)")
    
    print("\n" + "="*80)
    print("КОНВЕРТАЦИЯ ЗАВЕРШЕНА")
    print("="*80)
    print(f"\nФайлы готовы для обучения моделей:")
    print(f"  - {opensmile_output}: {len(opensmile_df)} сегментов, {opensmile_df['file_id'].nunique()} сессий")
    print(f"  - {text_output}: {len(text_df)} сегментов, {text_df['session_id'].nunique()} сессий")
    
    return opensmile_df, text_df


def main():
    parser = argparse.ArgumentParser(
        description='Конвертация формата из pipeline в формат для ML-моделей'
    )
    parser.add_argument(
        '--merged-features',
        type=str,
        default='data/features/merged_features.csv',
        help='Путь к merged_features.csv из pipeline'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DATA_DIR,
        help='Директория для сохранения результатов'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Перезаписать существующие файлы (вместо дополнения)'
    )
    
    args = parser.parse_args()
    
    convert_pipeline_format(
        args.merged_features,
        args.output_dir,
        args.overwrite
    )


if __name__ == '__main__':
    main()

