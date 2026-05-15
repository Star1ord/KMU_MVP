#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils import DATA_DIR
from processing.convert_pipeline_format import convert_pipeline_format
from training.build_multi_session_df import build_audio_session_df, build_text_session_df, merge_session_dfs
from src.utils.paths import (
    AUDIO_WAV_DIR,
    GENERAL_PIPELINE_DIR,
    OPENSMILE_DIR,
    PROCESSED_DATA_DIR,
    REPO_ROOT,
    TRANSCRIPTS_DIR,
    resolve_repo_path,
)


def _add_message(report: dict[str, Any], key: str, message: str) -> None:
    bucket = report.setdefault(key, [])
    if message not in bucket:
        bucket.append(message)


def _normalize_csv_path(value: str | Path) -> Path:
    return resolve_repo_path(value)


def copy_media_to_pipeline(media_files: list[str], label: int) -> tuple[list[str], dict[str, Path], dict[str, Path]]:
    label_dir = AUDIO_WAV_DIR / str(label)
    label_dir.mkdir(parents=True, exist_ok=True)

    file_ids: list[str] = []
    staged_media: dict[str, Path] = {}
    source_paths: dict[str, Path] = {}

    for media_file in media_files:
        source_path = Path(media_file).expanduser().resolve()
        if not source_path.exists():
            print(f"warning: file not found: {source_path}")
            continue

        file_id = source_path.stem
        target_path = label_dir / source_path.name

        for other_label in (0, 1):
            if other_label == label:
                continue

            other_dir = AUDIO_WAV_DIR / str(other_label)
            for conflicting_path in (
                other_dir / source_path.name,
                other_dir / f"{file_id}.wav",
            ):
                if conflicting_path.exists():
                    print(f"removing conflicting staged file: {conflicting_path}")
                    try:
                        conflicting_path.unlink()
                    except Exception as exc:  # pragma: no cover - best effort cleanup
                        print(f"warning: failed to remove {conflicting_path}: {exc}")

        if target_path.exists():
            try:
                same_file = source_path.samefile(target_path)
            except Exception:
                same_file = False

            if not same_file:
                print(f"removing existing staged file: {target_path}")
                target_path.unlink()

        if not target_path.exists():
            shutil.copy2(source_path, target_path)

        print(f"video uploaded: {target_path}")

        file_ids.append(file_id)
        staged_media[file_id] = target_path
        source_paths[file_id] = source_path

    return file_ids, staged_media, source_paths


def run_pipeline(
    file_ids: list[str],
    processed_dir: Path,
    whisper_model: str,
    whisper_device: str,
    opensmile_dir: Path,
    skip_transcription: bool = False,
    skip_segmentation: bool = False,
    skip_features: bool = False,
    skip_merge: bool = False,
) -> tuple[bool, Optional[str]]:
    print("\n" + "=" * 80)
    print("running pipeline")
    print("=" * 80)

    script_path = GENERAL_PIPELINE_DIR.parent / "run_full_pipeline.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--data-dir",
        str(processed_dir),
        "--file-ids",
        *file_ids,
        "--whisper-model",
        whisper_model,
        "--whisper-device",
        whisper_device,
        "--opensmile-dir",
        str(opensmile_dir),
        "--include-completed",
        "--yes",
    ]

    if skip_transcription:
        cmd.append("--skip-transcription")
    if skip_segmentation:
        cmd.append("--skip-segmentation")
    if skip_features:
        cmd.append("--skip-features")
    if skip_merge:
        cmd.append("--skip-merge")

    print(f"command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        print("pipeline completed successfully")
        return True, None
    except subprocess.CalledProcessError as exc:
        message = f"pipeline exited with code {exc.returncode}"
        print(f"error running pipeline: {message}")
        return False, message


def _load_csv_if_exists(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None

    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        print(f"warning: failed to read {csv_path}: {exc}")
        return None


def collect_pipeline_artifacts(
    report: dict[str, Any],
    file_ids: list[str],
    source_paths: dict[str, Path],
    label: int,
    processed_dir: Path,
    skip_transcription: bool,
    skip_segmentation: bool,
    skip_merge: bool,
) -> None:
    label_dir = AUDIO_WAV_DIR / str(label)
    report["wav_paths"] = {}
    report["transcript_paths"] = {}

    for file_id in file_ids:
        source_path = source_paths[file_id]
        expected_wav = label_dir / f"{file_id}.wav"
        report["wav_paths"][file_id] = str(expected_wav)

        if source_path.suffix.lower() != ".wav" and not expected_wav.exists():
            _add_message(
                report,
                "errors",
                f"WAV file was not created for session '{file_id}'. Expected: {expected_wav}",
            )
        elif expected_wav.exists():
            print(f"wav created: {expected_wav}")

        expected_transcript = processed_dir / "transcripts" / str(label) / f"{file_id}.json"
        report["transcript_paths"][file_id] = str(expected_transcript)

        if not skip_transcription and not expected_transcript.exists():
            _add_message(
                report,
                "errors",
                f"WhisperX output was not created for session '{file_id}'. Expected: {expected_transcript}",
            )

    segments_metadata_path = processed_dir / "segments" / "segments_metadata.csv"
    report["segments_metadata_path"] = str(segments_metadata_path)

    segments_df = _load_csv_if_exists(segments_metadata_path)
    if not skip_segmentation:
        if segments_df is None:
            _add_message(
                report,
                "errors",
                f"segments_metadata.csv was not created at {segments_metadata_path}",
            )
        else:
            matched_ids = set()
            for file_id in file_ids:
                session_rows = segments_df[segments_df["file_id"] == file_id]
                if session_rows.empty:
                    _add_message(
                        report,
                        "errors",
                        f"segments_metadata.csv exists but contains no rows for session '{file_id}'",
                    )
                    continue

                matched_ids.add(file_id)
                print(f"segments_metadata.csv created: {segments_metadata_path} ({len(session_rows)} rows for {file_id})")

            report["segments_found_for"] = sorted(matched_ids)

    merged_features_path = processed_dir / "features" / "merged_features.csv"
    report["merged_features_path"] = str(merged_features_path)

    merged_df = _load_csv_if_exists(merged_features_path)
    if not skip_merge:
        if merged_df is None:
            _add_message(
                report,
                "errors",
                f"Merged features file was not created at {merged_features_path}",
            )
        else:
            matched_ids = set()
            for file_id in file_ids:
                session_rows = merged_df[merged_df["file_id"] == file_id]
                if session_rows.empty:
                    _add_message(
                        report,
                        "errors",
                        f"merged_features.csv exists but contains no rows for session '{file_id}'",
                    )
                    continue

                matched_ids.add(file_id)

            report["merged_features_found_for"] = sorted(matched_ids)


def rebuild_multimodal_dataset(data_dir: str | Path = DATA_DIR) -> Optional[pd.DataFrame]:
    print("\n" + "=" * 80)
    print("rebuilding multimodal dataset")
    print("=" * 80)

    model_data_dir = _normalize_csv_path(data_dir)

    try:
        audio_path = model_data_dir / "opensmile_features.csv"
        text_path = model_data_dir / "text_features_nlp2_ready.csv"

        if not audio_path.exists() or not text_path.exists():
            print("error: required files not found after conversion")
            return None

        print("\n[1] building audio_session_df...")
        audio_session_df = build_audio_session_df(str(audio_path))

        print("\n[2] building text_session_df...")
        text_session_df = build_text_session_df(str(text_path))

        print("\n[3] merging datasets...")
        multi_session_df = merge_session_dfs(audio_session_df, text_session_df)

        output_path = model_data_dir / "multi_session_df.csv"
        multi_session_df.to_csv(output_path, index=False)
        print(f"multimodal dataset updated: {output_path}")
        print(f"total sessions: {len(multi_session_df)}")

        return multi_session_df

    except Exception as exc:
        print(f"error rebuilding dataset: {exc}")
        import traceback

        traceback.print_exc()
        return None


def process_new_media(
    media_files: list[str],
    label: int,
    pipeline_dir: str = str(GENERAL_PIPELINE_DIR.parent),
    pipeline_data_dir: str = str(PROCESSED_DATA_DIR),
    model_data_dir: str = DATA_DIR,
    whisper_model: str = "medium",
    whisper_device: str = "cpu",
    opensmile_dir: str = str(OPENSMILE_DIR),
    skip_pipeline: bool = False,
    add_to_training: bool = True,
    skip_transcription: bool = False,
    skip_segmentation: bool = False,
    skip_features: bool = False,
    skip_merge: bool = False,
    return_details: bool = False,
) -> bool | dict[str, Any]:
    del pipeline_dir  # kept for backward-compatible signature

    print("\n" + "=" * 80)
    print("processing new media files")
    print("=" * 80)
    print(f"files: {len(media_files)}")
    print(f"label: {label}")
    print(f"add to training dataset: {add_to_training}")

    report: dict[str, Any] = {
        "success": False,
        "file_ids": [],
        "staged_media_paths": {},
        "source_media_paths": {},
        "wav_paths": {},
        "transcript_paths": {},
        "segments_metadata_path": None,
        "merged_features_path": None,
        "errors": [],
        "warnings": [],
    }

    if not media_files:
        _add_message(report, "errors", "No files specified for processing")
        return report if return_details else False

    processed_dir = resolve_repo_path(pipeline_data_dir)
    model_dir = resolve_repo_path(model_data_dir)
    opensmile_path = resolve_repo_path(opensmile_dir)

    print("\n[step 1] copying files to pipeline...")
    file_ids, staged_paths, source_paths = copy_media_to_pipeline(media_files, label)

    report["file_ids"] = file_ids
    report["staged_media_paths"] = {file_id: str(path) for file_id, path in staged_paths.items()}
    report["source_media_paths"] = {file_id: str(path) for file_id, path in source_paths.items()}

    if not file_ids:
        _add_message(report, "errors", "Failed to copy media files into the pipeline staging area")
        return report if return_details else False

    print(f"copied {len(file_ids)} files: {file_ids}")

    pipeline_success = True
    pipeline_error: Optional[str] = None
    if not skip_pipeline:
        print("\n[step 2] processing through pipeline...")
        pipeline_success, pipeline_error = run_pipeline(
            file_ids=file_ids,
            processed_dir=processed_dir,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            opensmile_dir=opensmile_path,
            skip_transcription=skip_transcription,
            skip_segmentation=skip_segmentation,
            skip_features=skip_features,
            skip_merge=skip_merge,
        )
    else:
        print("\n[step 2] skipping pipeline processing (--skip-pipeline)")

    if pipeline_error:
        _add_message(report, "warnings", pipeline_error)

    collect_pipeline_artifacts(
        report=report,
        file_ids=file_ids,
        source_paths=source_paths,
        label=label,
        processed_dir=processed_dir,
        skip_transcription=skip_transcription,
        skip_segmentation=skip_segmentation,
        skip_merge=skip_merge,
    )

    if not pipeline_success and not report["errors"]:
        _add_message(report, "errors", pipeline_error or "Pipeline subprocess failed")

    if not add_to_training:
        report["success"] = pipeline_success and not report["errors"]
        print("\n" + "=" * 80)
        print("processing completed (not added to training dataset)")
        print("=" * 80)
        print(f"files processed: {len(file_ids)}")
        print("note: data processed through pipeline but not added to training dataset")
        print("ready for inference only")
        return report if return_details else report["success"]

    print("\n[step 3] converting format...")
    merged_features_path = Path(report["merged_features_path"]) if report["merged_features_path"] else None

    if merged_features_path is None or not merged_features_path.exists():
        _add_message(report, "errors", "Cannot convert pipeline format because merged_features.csv is missing")
        return report if return_details else False

    try:
        new_merged = pd.read_csv(merged_features_path)

        existing_merged_path = model_dir / "merged_features.csv"
        merged_input_path = merged_features_path
        if existing_merged_path.exists():
            existing_merged = pd.read_csv(existing_merged_path)
            new_file_ids = set(new_merged["file_id"].unique())
            existing_merged = existing_merged[~existing_merged["file_id"].isin(new_file_ids)]
            combined_merged = pd.concat([existing_merged, new_merged], ignore_index=True)
            combined_merged.to_csv(existing_merged_path, index=False)
            merged_input_path = existing_merged_path
            print(
                f"merged with existing data: {len(existing_merged)} + "
                f"{len(new_merged)} = {len(combined_merged)} segments"
            )

        convert_pipeline_format(str(merged_input_path), str(model_dir), overwrite=False)
    except Exception as exc:
        _add_message(report, "errors", f"Error converting pipeline format: {exc}")
        import traceback

        traceback.print_exc()
        return report if return_details else False

    print("\n[step 4] rebuilding multimodal dataset...")
    multi_df = rebuild_multimodal_dataset(model_dir)

    if multi_df is None:
        _add_message(report, "errors", "Failed to rebuild multimodal dataset")
        return report if return_details else False

    processed_session_ids: list[str] = []
    for file_id in file_ids:
        normalized_id = str(file_id).removeprefix("0_").removeprefix("1_")
        matching = multi_df[multi_df["session_id"].isin([file_id, normalized_id])]
        if matching.empty:
            matching = multi_df[multi_df["session_id"].astype(str).str.contains(normalized_id, na=False, regex=False)]
        if not matching.empty:
            processed_session_ids.extend(matching["session_id"].unique().tolist())

    report["processed_session_ids"] = processed_session_ids
    report["success"] = not report["errors"]

    print("\n" + "=" * 80)
    print("processing completed successfully")
    print("=" * 80)
    print("new sessions added to dataset:")
    print(f"  - files processed: {len(file_ids)}")
    print(f"  - sessions found in dataset: {len(processed_session_ids)}")
    if processed_session_ids:
        print(f"  - session IDs: {processed_session_ids}")
    print(f"  - total sessions in dataset: {len(multi_df)}")

    return report if return_details else report["success"]


def main() -> None:
    parser = argparse.ArgumentParser(description="process new audio/video files through pipeline")
    parser.add_argument("media_files", nargs="+")
    parser.add_argument("--label", type=int, required=True, choices=[0, 1])
    parser.add_argument("--pipeline-dir", type=str, default=str(GENERAL_PIPELINE_DIR.parent))
    parser.add_argument("--pipeline-data-dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--model-data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--whisper-model", type=str, default="medium")
    parser.add_argument("--whisper-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opensmile-dir", type=str, default=str(OPENSMILE_DIR))
    parser.add_argument("--skip-pipeline", action="store_true")

    args = parser.parse_args()

    success = process_new_media(
        media_files=args.media_files,
        label=args.label,
        pipeline_dir=args.pipeline_dir,
        pipeline_data_dir=args.pipeline_data_dir,
        model_data_dir=args.model_data_dir,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        opensmile_dir=args.opensmile_dir,
        skip_pipeline=args.skip_pipeline,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
