#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import DATA_DIR
from processing.convert_pipeline_format import convert_pipeline_format
from training.build_multi_session_df import build_audio_session_df, build_text_session_df, merge_session_dfs


def copy_media_to_pipeline(
    media_files: List[str],
    label: int,
    pipeline_data_dir: str = 'data'
) -> List[str]:
    audio_dir = Path('data/raw') / 'audio_wav' / str(label)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    file_ids = []
    base_audio_dir = Path('data/raw') / 'audio_wav'
    
    for media_file in media_files:
        media_path = Path(media_file)
        if not media_path.exists():
            print(f"warning: file not found: {media_file}")
            continue
        
        file_id = media_path.stem
        target_path = audio_dir / media_path.name
        
        # Проверяем, есть ли файл в других папках с другими labels
        for other_label in [0, 1]:
            if other_label != label:
                other_dir = base_audio_dir / str(other_label)
                other_path = other_dir / media_path.name
                if other_path.exists():
                    print(f"removing existing file from label {other_label}: {other_path}")
                    try:
                        other_path.unlink()
                    except Exception as e:
                        print(f"warning: failed to remove {other_path}: {e}")
        
        # Удаляем файл из целевой папки, если уже существует
        if target_path.exists():
            print(f"removing existing file from target directory: {target_path}")
            try:
                target_path.unlink()
            except Exception as e:
                print(f"warning: failed to remove {target_path}: {e}")
        
        print(f"copying: {media_path.name} -> {target_path}")
        shutil.copy2(media_path, target_path)
        
        file_ids.append(file_id)
    
    return file_ids


def run_pipeline(
    file_ids: List[str],
    pipeline_dir: str = 'my_pipeline',
    data_dir: str = 'data',
    whisper_model: str = 'medium',
    whisper_device: str = 'cpu',
    opensmile_dir: str = 'opensmile',
    skip_transcription: bool = False,
    skip_segmentation: bool = False,
    skip_features: bool = False,
    skip_merge: bool = False,
) -> bool:
    print("\n" + "="*80)
    print("running pipeline")
    print("="*80)
    
    cmd = [
        sys.executable,
        str(Path(pipeline_dir) / 'run_full_pipeline.py'),
        '--data-dir', data_dir,
        '--file-ids', *file_ids,
        '--whisper-model', whisper_model,
        '--whisper-device', whisper_device,
        '--opensmile-dir', opensmile_dir,
        '--include-completed'
    ]

    if skip_transcription:
        cmd.append('--skip-transcription')
    if skip_segmentation:
        cmd.append('--skip-segmentation')
    if skip_features:
        cmd.append('--skip-features')
    if skip_merge:
        cmd.append('--skip-merge')
    
    print(f"command: {' '.join(cmd)}\n")
    
    try:
        # If AUTO_YES is set in the environment, pass --yes to the pipeline script
        if os.getenv('AUTO_YES', '').lower() in ('1', 'true', 'yes'):
            cmd.append('--yes')
        result = subprocess.run(cmd, check=True)
        print("pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"error running pipeline: {e}")
        return False


def rebuild_multimodal_dataset(
    data_dir: str = DATA_DIR
) -> Optional[pd.DataFrame]:
    print("\n" + "="*80)
    print("rebuilding multimodal dataset")
    print("="*80)
    
    try:
        audio_path = os.path.join(data_dir, 'opensmile_features.csv')
        text_path = os.path.join(data_dir, 'text_features_nlp2_ready.csv')
        
        if not os.path.exists(audio_path) or not os.path.exists(text_path):
            print("error: required files not found after conversion")
            return None
        
        print("\n[1] building audio_session_df...")
        audio_session_df = build_audio_session_df(audio_path)
        
        print("\n[2] building text_session_df...")
        text_session_df = build_text_session_df(text_path)
        
        print("\n[3] merging datasets...")
        multi_session_df = merge_session_dfs(audio_session_df, text_session_df)
        
        output_path = os.path.join(data_dir, 'multi_session_df.csv')
        multi_session_df.to_csv(output_path, index=False)
        print(f"multimodal dataset updated: {output_path}")
        print(f"total sessions: {len(multi_session_df)}")
        
        return multi_session_df
        
    except Exception as e:
        print(f"error rebuilding dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_new_media(
    media_files: List[str],
    label: int,
    pipeline_dir: str = 'src/pipeline',
    pipeline_data_dir: str = 'data/processed',
    model_data_dir: str = DATA_DIR,
    whisper_model: str = 'medium',
    whisper_device: str = 'cpu',
    opensmile_dir: str = 'src/pipeline/opensmile',
    skip_pipeline: bool = False,
    add_to_training: bool = True,
    skip_transcription: bool = False,
    skip_segmentation: bool = False,
    skip_features: bool = False,
    skip_merge: bool = False,
) -> bool:
    print("\n" + "="*80)
    print("processing new media files")
    print("="*80)
    print(f"files: {len(media_files)}")
    print(f"label: {label}")
    print(f"add to training dataset: {add_to_training}")
    
    if not media_files:
        print("error: no files specified for processing")
        return False
    
    print("\n[step 1] copying files to pipeline...")
    file_ids = copy_media_to_pipeline(media_files, label, pipeline_data_dir)
    
    if not file_ids:
        print("error: failed to copy files")
        return False
    
    print(f"copied {len(file_ids)} files: {file_ids}")
    
    if not skip_pipeline:
        print("\n[step 2] processing through pipeline...")
        pipeline_success = run_pipeline(
            file_ids,
            pipeline_dir,
            pipeline_data_dir,
            whisper_model,
            whisper_device,
            opensmile_dir,
            skip_transcription=skip_transcription,
            skip_segmentation=skip_segmentation,
            skip_features=skip_features,
            skip_merge=skip_merge,
        )
        
        # Проверяем наличие финального файла, даже если pipeline вернул ошибку
        # (pipeline может завершиться с ошибкой, но данные все равно обработаться)
        merged_features_path = os.path.join(pipeline_data_dir, 'features', 'merged_features.csv')
        if not os.path.exists(merged_features_path):
            if not pipeline_success:
                print("error: pipeline failed and output file not found")
                return False
            else:
                print("warning: pipeline reported success but output file not found")
            return False
        elif not pipeline_success:
            print("warning: pipeline reported errors, but output file exists. continuing...")
    else:
        print("\n[step 2] skipping pipeline processing (--skip-pipeline)")
    
    # Если не добавляем в тренировочный датасет, завершаем здесь
    if not add_to_training:
        print("\n" + "="*80)
        print("processing completed (not added to training dataset)")
        print("="*80)
        print(f"files processed: {len(file_ids)}")
        print("note: data processed through pipeline but not added to training dataset")
        print("ready for inference only")
        return True
    
    print("\n[step 3] converting format...")
    merged_features_path = os.path.join(pipeline_data_dir, 'features', 'merged_features.csv')
    
    if not os.path.exists(merged_features_path):
        print(f"error: file not found: {merged_features_path}")
        return False
    
    try:
        import pandas as pd
        
        # Загружаем новые данные из pipeline
        new_merged = pd.read_csv(merged_features_path)
        
        # Загружаем существующие данные из model_data_dir, если есть
        existing_merged_path = os.path.join(model_data_dir, 'merged_features.csv')
        if os.path.exists(existing_merged_path):
            existing_merged = pd.read_csv(existing_merged_path)
            
            # Объединяем, исключая дубликаты по file_id
            new_file_ids = set(new_merged['file_id'].unique())
            existing_merged = existing_merged[~existing_merged['file_id'].isin(new_file_ids)]
            combined_merged = pd.concat([existing_merged, new_merged], ignore_index=True)
            
            # Сохраняем объединенный файл
            combined_merged.to_csv(existing_merged_path, index=False)
            print(f"merged with existing data: {len(existing_merged)} + {len(new_merged)} = {len(combined_merged)} segments")
            
            # Используем объединенный файл для конвертации
            merged_features_path = existing_merged_path
        
        convert_pipeline_format(
            merged_features_path,
            model_data_dir,
            overwrite=False
        )
    except Exception as e:
        print(f"error converting format: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[step 4] rebuilding multimodal dataset...")
    multi_df = rebuild_multimodal_dataset(model_data_dir)
    
    if multi_df is None:
        print("error: failed to rebuild dataset")
        return False
    
    print("\n" + "="*80)
    print("processing completed successfully")
    print("="*80)
    
    processed_session_ids = []
    for file_id in file_ids:
        import re
        def extract_base_file_id(fid):
            return re.sub(r'\s*\(\d+\)$', '', str(fid))
        def normalize_file_id(fid):
            fid_str = str(fid)
            if fid_str.startswith('0_'):
                fid_str = fid_str.replace('0_', '', 1)
            return extract_base_file_id(fid_str)
        
        normalized_id = normalize_file_id(file_id)
        matching = multi_df[multi_df['session_id'].isin([file_id, normalized_id])]
        if len(matching) == 0:
            base_id = file_id.split('_')[0] if '_' in file_id else file_id
            matching = multi_df[multi_df['session_id'].str.contains(base_id, na=False, regex=False)]
        if len(matching) > 0:
            processed_session_ids.extend(matching['session_id'].unique().tolist())
    
    print(f"new sessions added to dataset:")
    print(f"  - files processed: {len(file_ids)}")
    print(f"  - sessions found in dataset: {len(processed_session_ids)}")
    if processed_session_ids:
        print(f"  - session IDs: {processed_session_ids}")
    print(f"  - total sessions in dataset: {len(multi_df)}")
    print(f"ready for:")
    print(f"  - retraining models: python3 train_early_fusion.py")
    print(f"  - inference: streamlit run app.py")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='process new audio/video files through pipeline')
    parser.add_argument('media_files', nargs='+')
    parser.add_argument('--label', type=int, required=True, choices=[0, 1])
    parser.add_argument('--pipeline-dir', type=str, default='my_pipeline')
    parser.add_argument('--pipeline-data-dir', type=str, default='data')
    parser.add_argument('--model-data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--whisper-model', type=str, default='medium')
    parser.add_argument('--whisper-device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--opensmile-dir', type=str, default='opensmile')
    parser.add_argument('--skip-pipeline', action='store_true')
    
    args = parser.parse_args()
    
    success = process_new_media(
        args.media_files,
        args.label,
        args.pipeline_dir,
        args.pipeline_data_dir,
        args.model_data_dir,
        args.whisper_model,
        args.whisper_device,
        args.opensmile_dir,
        args.skip_pipeline
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

