from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

_project_root = Path(__file__).resolve().parent.parent.parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from pipeline.media_utils import discover_media


def run_command(cmd: List[str], description: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"command: {' '.join(cmd[:3])}...")
    print(f"{'=' * 60}")
    print(f"start: {time.strftime('%H:%M:%S')}")
    start_time = time.time()

    try:
        project_root = _project_root
        subprocess.run(cmd, check=True, cwd=str(project_root))
        elapsed = time.time() - start_time
        print(f"done: {description} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return True
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - start_time
        print(f"error: {description} failed (code {exc.returncode}, time: {elapsed:.1f}s)")
        return False


def check_dependencies() -> bool:
    required = {
        "python3": ["python3", "--version"],
        "ffmpeg": ["ffmpeg", "-version"],
    }
    all_ok = True
    for name, cmd in required.items():
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"ok: {name} found")
        except Exception:
            print(f"error: {name} not found")
            all_ok = False
    return all_ok


def load_completed_ids(merged_path: Path) -> set[str]:
    if not merged_path.exists():
        return set()

    completed: set[str] = set()
    try:
        with merged_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if not reader.fieldnames or "file_id" not in reader.fieldnames:
                return set()
            for row in reader:
                fid = row.get("file_id")
                if fid:
                    completed.add(fid)
    except Exception:  # pylint: disable=broad-except
        return set()
    return completed


def determine_targets(
    audio_dir: Path,
    requested_ids: List[str],
    limit: int | None,
    include_completed: bool,
    merged_path: Path,
    transcripts_root: Optional[Path] = None,
) -> List[str]:
    samples = discover_media(audio_dir, transcripts_root)
    available_ids = [sample.file_id for sample in samples]

    if requested_ids:
        missing = [fid for fid in requested_ids if fid not in available_ids]
        if missing:
            print(f"warning: {missing} not found in audio_wav")
        targets = [fid for fid in requested_ids if fid in available_ids]
    else:
        targets = available_ids

    if not include_completed:
        completed = load_completed_ids(merged_path)
        targets = [fid for fid in targets if fid not in completed]

    if limit is not None:
        targets = targets[:limit]

    return targets


def build_file_id_args(file_ids: List[str]) -> List[str]:
    if not file_ids:
        return []
    return ["--file-ids", *file_ids]


def main():
    parser = argparse.ArgumentParser(description="incremental pipeline run")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--file-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-completed", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-transcription", action="store_true")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--aggregate", action="store_true")

    parser.add_argument("--whisper-model", type=str, default="medium")
    parser.add_argument("--whisper-language", type=str, default="ru")
    parser.add_argument("--whisper-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--whisper-compute-type", type=str, default="int8")

    parser.add_argument("--min-segment", type=float, default=2.0)
    parser.add_argument("--max-segment", type=float, default=5.0)
    parser.add_argument("--opensmile-dir", type=str, default="src/pipeline/opensmile")
    parser.add_argument("--language", type=str, default="ru")

    args = parser.parse_args()

    if not check_dependencies():
        print("warning: missing dependencies. continue? (y/n)")
        if input().strip().lower() != "y":
            sys.exit(1)

    base_dir = Path(args.data_dir)
    audio_dir = Path("data/raw/audio_wav")
    transcript_dir = base_dir / "transcripts"
    segments_dir = base_dir / "segments"
    features_dir = base_dir / "features"
    merged_path = features_dir / "merged_features.csv"

    all_samples = discover_media(audio_dir, transcript_dir)
    all_ids = [s.file_id for s in all_samples]
    completed_ids = load_completed_ids(merged_path) if merged_path.exists() else set()
    
    print(f"\nstatistics:")
    print(f"   total files found: {len(all_ids)}")
    if completed_ids:
        print(f"   already processed: {len(completed_ids)}")
        print(f"   remaining: {len(set(all_ids) - completed_ids)}")
    else:
        print(f"   processed files: 0")
    
    target_ids = determine_targets(
        audio_dir,
        args.file_ids or [],
        args.limit if not args.file_ids else None,
        args.include_completed,
        merged_path,
        transcript_dir,
    )

    if not target_ids:
        print("\nwarning: nothing to process")
        if completed_ids and not args.include_completed:
            print(f"   all {len(all_ids)} files already processed")
            print(f"   use --include-completed to reprocess")
        else:
            print("   add new files to data/raw/audio_wav/0/ or data/raw/audio_wav/1/")
        return

    print(f"\nwill process ({len(target_ids)} files): {target_ids}")
    if args.limit and len(target_ids) < args.limit and not args.include_completed:
        skipped = len(set(all_ids) & completed_ids)
        if skipped > 0:
            print(f"   (skipped {skipped} already processed files. use --include-completed to include them)")
    
    steps = []
    if not args.skip_audio:
        steps.append("prepare audio")
    if not args.skip_transcription:
        steps.append("transcribe whisperx")
    if not args.skip_segmentation:
        steps.append("segment audio")
    if not args.skip_features:
        steps.append("extract opensmile features")
    if not args.skip_merge:
        steps.append("merge features")
    
    print(f"\nprocessing plan ({len(steps)} stages):")
    for i, step in enumerate(steps, 1):
        print(f"   {i}. {step}")
    
    file_id_args = build_file_id_args(target_ids)

    pipeline_dir = Path("pipeline") / "general_pipeline"
    python_cmd = sys.executable
    success = True
    current_step = 0

    if success and not args.skip_audio:
        current_step += 1
        cmd = [
            python_cmd,
            str(pipeline_dir / "extract_audio.py"),
            "--input-dir",
            str(audio_dir),
        ] + file_id_args
        success = run_command(cmd, f"[{current_step}/{len(steps)}] prepare audio")

    if success and not args.skip_transcription:
        current_step += 1
        cmd = [
            python_cmd,
            str(pipeline_dir / "transcribe_whisperx.py"),
            "--input-dir",
            str(audio_dir),
            "--output-dir",
            str(transcript_dir),
            "--model",
            args.whisper_model,
            "--language",
            args.whisper_language,
            "--device",
            args.whisper_device,
            "--compute-type",
            args.whisper_compute_type,
        ] + file_id_args
        success = run_command(cmd, f"[{current_step}/{len(steps)}] transcribe whisperx")

    if success and not args.skip_segmentation:
        current_step += 1
        cmd = [
            python_cmd,
            str(pipeline_dir / "segment_audio.py"),
            "--audio-dir",
            str(audio_dir),
            "--transcript-dir",
            str(transcript_dir),
            "--output-dir",
            str(segments_dir),
            "--min-duration",
            str(args.min_segment),
            "--max-duration",
            str(args.max_segment),
        ] + file_id_args
        success = run_command(cmd, f"[{current_step}/{len(steps)}] segment audio")

    if success and not args.skip_features:
        current_step += 1
        cmd = [
            python_cmd,
            str(pipeline_dir / "extract_opensmile_features.py"),
            "--segments-dir",
            str(segments_dir),
            "--output-dir",
            str(features_dir),
            "--opensmile-dir",
            args.opensmile_dir,
        ] + file_id_args
        success = run_command(cmd, f"[{current_step}/{len(steps)}] extract opensmile features")

    if success and not args.skip_merge:
        current_step += 1
        cmd = [
            python_cmd,
            str(pipeline_dir / "merge_features.py"),
            "--segments-metadata",
            str(segments_dir / "segments_metadata.csv"),
            "--opensmile-features",
            str(features_dir / "opensmile_features.csv"),
            "--output",
            str(features_dir / "merged_features.csv"),
            "--language",
            args.language,
            "--audio-dir",
            str(audio_dir),
        ] + file_id_args
        if args.limit and not args.file_ids:
            cmd.extend(["--limit", str(args.limit)])
        if args.aggregate:
            cmd.append("--aggregate")
        success = run_command(cmd, f"[{current_step}/{len(steps)}] merge features")

    print("\n" + "=" * 60)
    if success:
        print("pipeline completed")
        print(f"features file: {features_dir / 'merged_features.csv'}")
        print("run visualization: streamlit run visualization_app/app.py")
    else:
        print("pipeline completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()

