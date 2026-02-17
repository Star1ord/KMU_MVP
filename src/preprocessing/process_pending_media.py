#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Set
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import DATA_DIR
from processing.convert_pipeline_format import convert_pipeline_format
from training.build_multi_session_df import build_audio_session_df, build_text_session_df, merge_session_dfs


def get_processed_file_ids(pipeline_data_dir: str = 'data/processed') -> Set[str]:
    possible_paths = [
        os.path.join(pipeline_data_dir, 'features', 'merged_features.csv'),
        os.path.join('data_ml', 'merged_features.csv'),
    ]
    
    for merged_features_path in possible_paths:
        if os.path.exists(merged_features_path):
            try:
                df = pd.read_csv(merged_features_path)
                if 'file_id' in df.columns:
                    file_ids = set(df['file_id'].unique())
                    normalized_ids = set()
                    for fid in file_ids:
                        clean_id = str(fid).replace('0_', '', 1) if str(fid).startswith('0_') else str(fid)
                        normalized_ids.add(clean_id)
                        normalized_ids.add(str(fid))
                    return normalized_ids
            except Exception:
                continue
    
    return set()


def find_new_media_files(
    audio_wav_dir: str = 'data/audio_wav',
    processed_ids: Set[str] = None
) -> List[tuple[str, int]]:
    if processed_ids is None:
        processed_ids = set()
    
    audio_wav_path = Path(audio_wav_dir)
    new_files = []
    
    media_extensions = {'.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.m4a'}
    
    for label in [0, 1]:
        label_dir = audio_wav_path / str(label)
        
        if not label_dir.exists():
            continue
        
        for media_file in label_dir.iterdir():
            if not media_file.is_file():
                continue
            
            if media_file.suffix.lower() not in media_extensions:
                continue
            
            file_id = media_file.stem
            
            file_id_with_prefix = f"{label}_{file_id}" if not file_id.startswith(f"{label}_") else file_id
            
            if file_id not in processed_ids and file_id_with_prefix not in processed_ids:
                new_files.append((str(media_file), label))
    
    return new_files


def delete_processed_files(
    file_ids: List[str],
    audio_wav_dir: str = 'data/audio_wav'
) -> None:
    audio_wav_path = Path(audio_wav_dir)
    media_extensions = {'.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.m4a'}
    
    deleted_count = 0
    
    for label in [0, 1]:
        label_dir = audio_wav_path / str(label)
        
        if not label_dir.exists():
            continue
        
        for media_file in label_dir.iterdir():
            if not media_file.is_file():
                continue
            
            if media_file.suffix.lower() not in media_extensions:
                continue
            
            file_id = media_file.stem
            
            if file_id in file_ids:
                try:
                    media_file.unlink()
                    deleted_count += 1
                    print(f"deleted: {media_file.name}")
                except Exception as e:
                    print(f"warning: failed to delete {media_file.name}: {e}")
    
    if deleted_count > 0:
        print(f"deleted {deleted_count} processed files")


def process_pending_media(
    audio_wav_dir: str = 'data/audio_wav',
    pipeline_dir: str = 'my_pipeline',
    pipeline_data_dir: str = 'data/processed',
    model_data_dir: str = DATA_DIR,
    whisper_model: str = 'medium',
    whisper_device: str = 'cpu',
    opensmile_dir: str = 'src/pipeline/opensmile',
    delete_after_processing: bool = True,
    limit: int = None
) -> bool:
    print("\n" + "="*80)
    print("automatic processing of new media files")
    print("="*80)
    
    print("\n[step 1] finding already processed files...")
    processed_ids = get_processed_file_ids(pipeline_data_dir)
    print(f"already processed: {len(processed_ids)} files")
    
    print("\n[step 2] finding new files in folders...")
    new_files = find_new_media_files(audio_wav_dir, processed_ids)
    
    if not new_files:
        print("no new files to process")
        return True
    
    files_by_label = {0: [], 1: []}
    for file_path, label in new_files:
        files_by_label[label].append(file_path)
    
    print(f"new files found:")
    print(f"  control (0): {len(files_by_label[0])}")
    print(f"  risk (1): {len(files_by_label[1])}")
    print(f"  total: {len(new_files)}")
    
    if limit is not None:
        total_files = sum(len(files) for files in files_by_label.values())
        if total_files > limit:
            print(f"warning: limit reached, processing only {limit} files")
            all_files_flat = []
            for label in [0, 1]:
                all_files_flat.extend([(f, label) for f in files_by_label[label]])
            new_files = all_files_flat[:limit]
            files_by_label = {0: [], 1: []}
            for file_path, label in new_files:
                files_by_label[label].append(file_path)
    
    if not new_files:
        print("no files to process")
        return True
    
    all_processed_file_ids = []
    total_start_time = time.time()
    
    for label in [0, 1]:
        if not files_by_label[label]:
            continue
        
        print(f"\n[step 3] processing group {label}: {len(files_by_label[label])} files")
        
        cmd = [
            'python3',
            str(Path(pipeline_dir) / 'run_full_pipeline.py'),
            '--data-dir', pipeline_data_dir,
            '--file-ids', *[Path(f).stem for f in files_by_label[label]],
            '--whisper-model', whisper_model,
            '--whisper-device', whisper_device,
            '--opensmile-dir', opensmile_dir,
        ]
        
        print(f"command: {' '.join(cmd[:5])} ... {len(files_by_label[label])} files")
        print(f"starting processing at {time.strftime('%H:%M:%S')}...\n")
        step_start_time = time.time()
        
        try:
            result = subprocess.run(cmd, check=True)
            step_elapsed = time.time() - step_start_time
            print(f"group {label} processed successfully in {step_elapsed:.1f}s ({step_elapsed/60:.1f} min)")
            
            for file_path in files_by_label[label]:
                all_processed_file_ids.append(Path(file_path).stem)
                
        except subprocess.CalledProcessError as e:
            step_elapsed = time.time() - step_start_time
            print(f"error processing group {label} (time: {step_elapsed:.1f}s)")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"stderr: {e.stderr[:500]}")
            continue
    
    if not all_processed_file_ids:
        print("error: failed to process any files")
        return False
    
    print(f"\n[step 4] converting format...")
    merged_features_path = os.path.join(pipeline_data_dir, 'features', 'merged_features.csv')
    
    if not os.path.exists(merged_features_path):
        alt_path = os.path.join('data', 'ml', 'merged_features.csv')
        if os.path.exists(alt_path):
            merged_features_path = alt_path
        else:
            print(f"error: merged_features.csv not found")
            return False
    
    try:
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
    
    print(f"\n[step 5] rebuilding multimodal dataset...")
    multi_session_df = None
    try:
        audio_path = os.path.join(model_data_dir, 'opensmile_features.csv')
        text_path = os.path.join(model_data_dir, 'text_features_nlp2_ready.csv')
        
        if not os.path.exists(audio_path) or not os.path.exists(text_path):
            print("error: required files not found after conversion")
            return False
        
        audio_session_df = build_audio_session_df(audio_path)
        text_session_df = build_text_session_df(text_path)
        multi_session_df = merge_session_dfs(audio_session_df, text_session_df)
        
        output_path = os.path.join(model_data_dir, 'multi_session_df.csv')
        multi_session_df.to_csv(output_path, index=False)
        print(f"dataset updated: {len(multi_session_df)} sessions")
        
    except Exception as e:
        print(f"error rebuilding dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if delete_after_processing:
        print(f"\n[step 6] deleting processed files...")
        delete_processed_files(all_processed_file_ids, audio_wav_dir)
    
    total_elapsed = time.time() - total_start_time
    print("\n" + "="*80)
    print("processing completed successfully")
    print("="*80)
    print(f"files processed: {len(all_processed_file_ids)}")
    if multi_session_df is not None:
        print(f"total sessions in dataset: {len(multi_session_df)}")
    print(f"total execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    
    if delete_after_processing:
        print(f"processed files deleted from folders")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='automatic processing of new media files from data/audio_wav/0/ and data/audio_wav/1/')
    parser.add_argument('--audio-wav-dir', type=str, default='data/audio_wav')
    parser.add_argument('--pipeline-dir', type=str, default='my_pipeline')
    parser.add_argument('--pipeline-data-dir', type=str, default='data')
    parser.add_argument('--model-data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--whisper-model', type=str, default='medium')
    parser.add_argument('--whisper-device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--opensmile-dir', type=str, default='src/pipeline/opensmile')
    parser.add_argument('--no-delete', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    
    args = parser.parse_args()
    
    success = process_pending_media(
        args.audio_wav_dir,
        args.pipeline_dir,
        args.pipeline_data_dir,
        args.model_data_dir,
        args.whisper_model,
        args.whisper_device,
        args.opensmile_dir,
        delete_after_processing=not args.no_delete,
        limit=args.limit
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

