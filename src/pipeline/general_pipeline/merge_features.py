import os
import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import numpy as np

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.media_utils import map_file_labels


def calculate_speech_rate(text: str, duration: float) -> float:
    # speed word/sec
    if duration <= 0:
        return 0.0
    
    words = text.split()
    return len(words) / duration


def calculate_articulation_rate(text: str, duration: float, pause_ratio: float = 0.0) -> float:
    # articulation rate word/sec
    if duration <= 0:
        return 0.0
    
    words = text.split()
    active_duration = duration * (1 - pause_ratio)
    
    if active_duration <= 0:
        return 0.0
    
    return len(words) / active_duration


def calculate_type_token_ratio(text: str) -> float:
    # lexical diversity
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    
    unique_words = len(set(words))
    return unique_words / len(words)


def count_pronouns(text: str, language: str = 'ru') -> int:
    if language == 'ru':
        pronouns = [
            'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
            'меня', 'тебя', 'его', 'её', 'нас', 'вас', 'их',
            'мне', 'тебе', 'ему', 'ей', 'нам', 'вам', 'им',
            'мной', 'тобой', 'им', 'ей', 'нами', 'вами', 'ими',
            'мой', 'твой', 'его', 'её', 'наш', 'ваш', 'их'
        ]
    else:
        pronouns = [
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'our', 'their'
        ]
    
    words = text.lower().split()
    return sum(1 for word in words if word in pronouns)


def count_negations(text: str, language: str = 'ru') -> int:
    if language == 'ru':
        negations = ['не', 'нет', 'ни', 'никогда', 'ничего', 'никто']
    else:
        negations = ['not', 'no', 'never', 'nothing', 'nobody', 'none']
    
    words = text.lower().split()
    return sum(1 for word in words if word in negations)


def calculate_pause_ratio_from_words(words: list, segment_start: float, segment_end: float) -> float:
    """
    Вычисляет долю пауз в сегменте на основе временных меток слов.
    
    Args:
        words: Список словарей с ключами 'start' и 'end' для каждого слова
        segment_start: Начало сегмента
        segment_end: Конец сегмента
    
    Returns:
        Доля времени, занятая паузами (0.0 - 1.0)
    """
    if not words or segment_end <= segment_start:
        return 0.0
    
    total_duration = segment_end - segment_start
    
    # Вычисляем общее время речи (сумма длительностей всех слов)
    speech_duration = 0.0
    for word in words:
        if 'start' in word and 'end' in word:
            word_duration = word['end'] - word['start']
            if word_duration > 0:
                speech_duration += word_duration
    
    # Время пауз = общая длительность - время речи
    pause_duration = max(0.0, total_duration - speech_duration)
    
    # Доля пауз
    pause_ratio = pause_duration / total_duration if total_duration > 0 else 0.0
    
    return min(1.0, max(0.0, pause_ratio))  # Ограничиваем диапазон [0, 1]


def add_pause_ratios_from_transcripts(df: pd.DataFrame, transcripts_dir: str = 'data/processed/transcripts') -> pd.DataFrame:
    """
    Добавляет столбец pause_ratio в датафрейм на основе JSON транскрипций.
    
    Args:
        df: DataFrame с колонками file_id, segment_id, start, end
        transcripts_dir: Директория с JSON транскрипциями
    
    Returns:
        DataFrame с добавленным столбцом pause_ratio
    """
    df = df.copy()
    df['pause_ratio'] = 0.0
    
    transcripts_path = Path(transcripts_dir)
    
    # Загружаем транскрипции для каждого file_id
    for file_id in df['file_id'].unique():
        # Определяем label из file_id или из директории
        # Ищем JSON файлы в обеих поддиректориях (0/ и 1/)
        json_file = None
        for label_dir in ['0', '1']:
            potential_path = transcripts_path / label_dir / f"{file_id}.json"
            if potential_path.exists():
                json_file = potential_path
                break
        
        if not json_file or not json_file.exists():
            print(f"Warning: transcript not found for {file_id}, using pause_ratio=0")
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
            
            # Обрабатываем каждый сегмент
            for seg in transcript.get('segments', []):
                seg_start = seg.get('start', 0)
                seg_end = seg.get('end', 0)
                words = seg.get('words', [])
                
                if not words:
                    continue
                
                # Вычисляем pause_ratio для этого сегмента
                pause_ratio = calculate_pause_ratio_from_words(words, seg_start, seg_end)
                
                # Обновляем в df для соответствующего segment_id
                # Находим строку с тем же file_id и близкими start/end
                mask = (
                    (df['file_id'] == file_id) &
                    (abs(df['start'] - seg_start) < 0.01) &
                    (abs(df['end'] - seg_end) < 0.01)
                )
                
                df.loc[mask, 'pause_ratio'] = pause_ratio
        
        except Exception as e:
            print(f"Error loading transcript for {file_id}: {e}")
            continue
    
    return df


def add_text_features(df: pd.DataFrame, language: str = 'ru') -> pd.DataFrame:
    df = df.copy()
    
    df['speech_rate'] = df.apply(
        lambda row: calculate_speech_rate(
            str(row.get('text', '')),
            float(row.get('duration', 0))
        ),
        axis=1
    )
    
    df['articulation_rate'] = df.apply(
        lambda row: calculate_articulation_rate(
            str(row.get('text', '')),
            float(row.get('duration', 0)),
            float(row.get('pause_ratio', 0))
        ),
        axis=1
    )
    
    df['type_token_ratio'] = df['text'].apply(calculate_type_token_ratio)
    df['pronoun_count'] = df['text'].apply(lambda x: count_pronouns(str(x), language))
    df['negation_count'] = df['text'].apply(lambda x: count_negations(str(x), language))
    
    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0.0
    )
    
    return df


def filter_by_file_ids(df: pd.DataFrame, file_ids: Optional[Sequence[str]]) -> pd.DataFrame:
    if not file_ids:
        return df
    allowed = {fid.strip() for fid in file_ids if fid.strip()}
    if not allowed:
        return df
    return df[df['file_id'].isin(allowed)].copy()


def limit_file_ids(df: pd.DataFrame, limit: Optional[int]) -> pd.DataFrame:
    if limit is None or limit <= 0:
        return df
    ordered_ids = df['file_id'].drop_duplicates().head(limit)
    return df[df['file_id'].isin(ordered_ids)].copy()


def ensure_label_column(df: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    df = df.copy()
    
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
    
    if 'audio_path' in df.columns:
        def from_path(path_value: str) -> Optional[int]:
            if pd.isna(path_value):
                return None
            try:
                for part in Path(str(path_value)).parts:
                    if part in {'0', '1'}:
                        return int(part)
            except Exception:
                pass
            return None

        if 'label' in df.columns:
            mask = df['label'].isna()
        else:
            mask = pd.Series([True] * len(df), index=df.index)
        if mask.any():
            path_labels = df.loc[mask, 'audio_path'].apply(from_path)
            if 'label' not in df.columns:
                df['label'] = None
            df.loc[mask, 'label'] = pd.to_numeric(path_labels, errors='coerce')

    if 'label' not in df.columns or df['label'].isna().any():
        label_map = map_file_labels(audio_dir)
        if 'label' not in df.columns:
            df['label'] = None
        mask = df['label'].isna()
        if mask.any():
            df.loc[mask, 'label'] = df.loc[mask, 'file_id'].map(label_map)

    if 'label' not in df.columns or df['label'].isna().any():
        transcripts_root = audio_dir.parent / 'transcripts'
        label_map_from_transcripts = {}
        
        for label_val in [0, 1]:
            label_dir = transcripts_root / str(label_val)
            if label_dir.exists():
                for transcript_file in label_dir.glob('*.json'):
                    file_id = transcript_file.stem
                    file_id_clean = file_id.replace('0_', '', 1) if file_id.startswith('0_') else file_id
                    label_map_from_transcripts[file_id] = label_val
                    label_map_from_transcripts[file_id_clean] = label_val
        
        if 'label' not in df.columns:
            df['label'] = None
        mask = df['label'].isna()
        if mask.any():
            def get_label_from_transcripts(file_id: str) -> Optional[int]:
                if file_id in label_map_from_transcripts:
                    return label_map_from_transcripts[file_id]
                file_id_clean = file_id.replace('0_', '', 1) if file_id.startswith('0_') else file_id
                if file_id_clean in label_map_from_transcripts:
                    return label_map_from_transcripts[file_id_clean]
                return None
            
            transcript_labels = df.loc[mask, 'file_id'].apply(get_label_from_transcripts)
            df.loc[mask, 'label'] = pd.to_numeric(transcript_labels, errors='coerce')

    if 'label' not in df.columns or df['label'].isna().any():
        missing = df[df['label'].isna()]['file_id'].unique()
        print(f"warning: could not determine label for files: {sorted(missing)}")
        print(f"these files will be labeled as 0. add them to data/raw/audio_wav/0/ or data/raw/audio_wav/1/")

    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    return df


def merge_features(
    segments_metadata_path: str,
    opensmile_features_path: str,
    output_path: str = 'data/processed/features/merged_features.csv',
    language: str = 'ru',
    file_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    audio_dir: str = 'data/raw/audio_wav',
) -> pd.DataFrame:
    segments_path = Path(segments_metadata_path)
    opensmile_path = Path(opensmile_features_path)
    output_csv_path = Path(output_path)

    if not segments_path.exists():
        raise FileNotFoundError(
            f"segments_metadata.csv not found at {segments_path}. "
            "WhisperX transcription or audio segmentation likely failed."
        )
    if not opensmile_path.exists():
        raise FileNotFoundError(
            f"openSMILE features file not found at {opensmile_path}. "
            "Segment feature extraction likely failed."
        )

    segments_df = pd.read_csv(segments_path)
    opensmile_df = pd.read_csv(opensmile_path)

    segments_df = filter_by_file_ids(segments_df, file_ids)
    opensmile_df = filter_by_file_ids(opensmile_df, file_ids)

    segments_df = limit_file_ids(segments_df, limit)
    opensmile_df = limit_file_ids(opensmile_df, limit)

    available_ids = set(segments_df['file_id']).intersection(set(opensmile_df['file_id']))
    if not available_ids:
        print("warning: no file_id intersection between segments and features")
        if output_csv_path.exists():
            return pd.read_csv(output_csv_path)
        return pd.DataFrame()

    def normalize_file_id(file_id: str) -> str:
        if isinstance(file_id, str) and file_id.startswith('0_'):
            return file_id.replace('0_', '', 1)
        return str(file_id)
    
    segments_df = segments_df.copy()
    opensmile_df = opensmile_df.copy()
    segments_df['file_id_normalized'] = segments_df['file_id'].apply(normalize_file_id)
    opensmile_df['file_id_normalized'] = opensmile_df['file_id'].apply(normalize_file_id)
    
    available_ids_normalized = set(segments_df['file_id_normalized']).intersection(set(opensmile_df['file_id_normalized']))
    if not available_ids_normalized:
        print("warning: no file_id intersection after normalization")
        if output_csv_path.exists():
            return pd.read_csv(output_csv_path)
        return pd.DataFrame()
    
    segments_df = segments_df[segments_df['file_id_normalized'].isin(available_ids_normalized)]
    opensmile_df = opensmile_df[opensmile_df['file_id_normalized'].isin(available_ids_normalized)]
    
    merged_df = segments_df.merge(
        opensmile_df,
        left_on=['segment_id', 'file_id_normalized'],
        right_on=['segment_id', 'file_id_normalized'],
        how='inner',
        suffixes=('', '_opensmile')
    )
    
    if 'file_id_opensmile' in merged_df.columns:
        merged_df = merged_df.drop(columns=['file_id_opensmile'])
    merged_df = merged_df.drop(columns=['file_id_normalized'])

    merged_df = ensure_label_column(merged_df, Path(audio_dir))
    
    # Добавляем pause_ratio из JSON транскрипций перед вычислением текстовых признаков
    merged_df = add_pause_ratios_from_transcripts(merged_df)
    
    merged_df = add_text_features(merged_df, language=language)

    if output_csv_path.exists():
        existing_merged = pd.read_csv(output_csv_path)
        new_file_ids = set(merged_df['file_id'].unique())
        existing_merged = existing_merged[~existing_merged['file_id'].isin(new_file_ids)]
        merged_df = pd.concat([existing_merged, merged_df], ignore_index=True)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv_path, index=False, encoding='utf-8')

    print(f"merged dataset saved to {output_csv_path}")
    print(f"total rows: {len(merged_df)}")
    print(f"total features: {len(merged_df.columns)}")
    print(f"unique videos: {merged_df['file_id'].nunique()}")

    return merged_df


def aggregate_per_video(
    merged_features_path: str,
    output_path: str = 'data/features/features_per_video.csv'
) -> pd.DataFrame:
    df = pd.read_csv(merged_features_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['segment_id', 'start', 'end', 'duration']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'std', 'min', 'max', 'median']
    
    grouped = df.groupby('file_id')[numeric_cols].agg(agg_dict)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    if 'label' in df.columns:
        labels = df.groupby('file_id')['label'].first()
        grouped['label'] = labels
    
    grouped = grouped.reset_index()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"aggregated dataset saved to {output_path}")
    print(f"total videos: {len(grouped)}")
    
    return grouped


def main():
    parser = argparse.ArgumentParser(description='merge audio and text features')
    parser.add_argument('--segments-metadata', type=str, default='data/processed/segments/segments_metadata.csv')
    parser.add_argument('--opensmile-features', type=str, default='data/processed/features/opensmile_features.csv')
    parser.add_argument('--output', type=str, default='data/processed/features/merged_features.csv')
    parser.add_argument('--language', type=str, default='ru')
    parser.add_argument('--audio-dir', type=str, default='data/raw/audio_wav')
    parser.add_argument('--file-ids', nargs='+', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--aggregate', action='store_true')
    
    args = parser.parse_args()
    
    merged_df = merge_features(
        args.segments_metadata,
        args.opensmile_features,
        args.output,
        language=args.language,
        file_ids=args.file_ids,
        limit=args.limit,
        audio_dir=args.audio_dir
    )
    
    if args.aggregate:
        aggregate_per_video(args.output)


if __name__ == '__main__':
    main()
