import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline.media_utils import (
    MediaSample,
    discover_media,
    filter_samples,
    migrate_transcript,
)

SEGMENTS_METADATA_FILE = "segments_metadata.csv"


def load_transcript(transcript_path: Path) -> Optional[dict]:
    try:
        with open(transcript_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError:
        return None


def resolve_transcript(sample: MediaSample, transcripts_root: Path) -> Optional[Path]:
    migrate_transcript(sample, transcripts_root)
    new_path = sample.transcript_path(transcripts_root)
    if new_path.exists():
        return new_path

    legacy_path = sample.legacy_transcript_path(transcripts_root)
    if legacy_path.exists():
        return legacy_path

    return None


def create_segments_from_transcript(
    sample: MediaSample,
    transcript: dict,
    audio_path: Path,
    output_dir: Path,
    transcript_rel_path: Path,
    min_segment_duration: float,
    max_segment_duration: float,
) -> pd.DataFrame:
    audio = AudioSegment.from_wav(audio_path)
    file_id = sample.file_id
    audio_rel_path = Path("audio_wav") / sample.audio_subpath()

    segments_metadata = []
    segment_idx = 0

    if "segments" not in transcript:
        return pd.DataFrame()

    for seg in transcript["segments"]:
        start_time = seg["start"]
        end_time = seg["end"]
        text = seg.get("text", "").strip()
        duration = end_time - start_time

        if duration < min_segment_duration:
            continue

        def save_segment(sub_start: float, sub_end: float, words):
            segment_file = output_dir / f"{segment_idx:04d}.wav"
            start_ms = int(sub_start * 1000)
            end_ms = int(sub_end * 1000)
            segment_audio = audio[start_ms:end_ms]
            segment_audio.export(segment_file, format="wav")

            word_confidences = [w.get("score", 0.0) for w in words]
            segments_metadata.append(
                {
                    "file_id": file_id,
                    "label": sample.label,
                    "segment_id": f"{file_id}_segment_{segment_idx:04d}",
                    "audio_path": str(audio_rel_path),
                    "transcript_path": str(transcript_rel_path),
                    "segment_path": str(segment_file.relative_to(output_dir.parent)),
                    "start": sub_start,
                    "end": sub_end,
                    "duration": sub_end - sub_start,
                    "text": " ".join(word.get("word", "") for word in words) or text,
                    "asr_conf_mean": sum(word_confidences) / len(word_confidences)
                    if word_confidences
                    else 0.0,
                    "asr_conf_std": pd.Series(word_confidences).std()
                    if len(word_confidences) > 1
                    else 0.0,
                    "word_count": len(words),
                }
            )

        words = seg.get("words", [])

        if duration > max_segment_duration:
            num_subsegments = int(duration / max_segment_duration) + 1
            sub_duration = duration / num_subsegments

            for i in range(num_subsegments):
                sub_start = start_time + i * sub_duration
                sub_end = min(start_time + (i + 1) * sub_duration, end_time)
                segment_words = [
                    w
                    for w in words
                    if (sub_start <= w.get("start", 0) < sub_end)
                    or (sub_start < w.get("end", 0) <= sub_end)
                ]
                save_segment(sub_start, sub_end, segment_words)
                segment_idx += 1
        else:
            save_segment(start_time, end_time, words)
            segment_idx += 1

    return pd.DataFrame(segments_metadata)


def determine_targets(
    samples: List[MediaSample],
    metadata_path: Path,
    file_ids: Optional[List[str]],
    limit: Optional[int],
    force: bool,
) -> List[MediaSample]:
    selected = filter_samples(samples, file_ids)

    processed_ids: set[str] = set()
    if metadata_path.exists():
        try:
            existing_df = pd.read_csv(metadata_path, usecols=["file_id"])
            processed_ids = set(existing_df["file_id"].unique())
        except Exception:
            processed_ids = set()

    tasks = []
    for sample in selected:
        if not sample.wav_path.exists():
            print(f"warning: skipping {sample.file_id}: wav not found ({sample.wav_path})")
            continue
        if not force and sample.file_id in processed_ids:
            continue
        tasks.append(sample)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def merge_metadata(existing_path: Path, new_df: pd.DataFrame, processed_ids: List[str]) -> None:
    if existing_path.exists():
        try:
            existing_df = pd.read_csv(existing_path)
            # Проверяем, что файл не пустой
            if len(existing_df) > 0 and len(existing_df.columns) > 0:
                existing_df = existing_df[~existing_df["file_id"].isin(processed_ids)]
                combined = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Файл пустой или поврежден, используем только новые данные
                combined = new_df
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            # Файл поврежден или пустой, используем только новые данные
            print(f"Warning: {existing_path} is empty or corrupted, using only new data: {e}")
            combined = new_df
    else:
        combined = new_df

    combined.to_csv(existing_path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="segment audio by transcripts")
    parser.add_argument("--audio-dir", type=str, default="data/raw/audio_wav")
    parser.add_argument("--transcript-dir", type=str, default="data/processed/transcripts")
    parser.add_argument("--output-dir", type=str, default="data/processed/segments")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--file-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    audio_root = Path(args.audio_dir)
    transcripts_root = Path(args.transcript_dir)
    segments_root = Path(args.output_dir)
    metadata_path = segments_root / SEGMENTS_METADATA_FILE

    samples = discover_media(audio_root)
    if not samples:
        print(f"warning: no audio found in {audio_root}")
        return

    targets = determine_targets(
        samples,
        metadata_path,
        file_ids=args.file_ids,
        limit=args.limit,
        force=args.force,
    )

    if not targets:
        print("no files to segment")
        return

    all_segments = []
    processed_ids = []

    for sample in tqdm(targets, desc="segmenting"):
        transcript_path = resolve_transcript(sample, transcripts_root)
        if transcript_path is None:
            print(f"warning: skipping {sample.file_id}: transcript not found")
            continue

        transcript = load_transcript(transcript_path)
        if not transcript:
            print(f"warning: skipping {sample.file_id}: empty transcript")
            continue

        segment_dir = segments_root / sample.file_id
        if args.force and segment_dir.exists():
            shutil.rmtree(segment_dir)
        segment_dir.mkdir(parents=True, exist_ok=True)

        transcript_rel = Path("transcripts") / str(sample.label) / f"{sample.file_id}.json"
        segments_df = create_segments_from_transcript(
            sample,
            transcript,
            sample.wav_path,
            segment_dir,
            transcript_rel,
            args.min_duration,
            args.max_duration,
        )

        if not segments_df.empty:
            all_segments.append(segments_df)
            processed_ids.append(sample.file_id)
        else:
            print(f"warning: {sample.file_id}: failed to create segments")

    if not all_segments:
        print("warning: no segments created")
        return

    combined_df = pd.concat(all_segments, ignore_index=True)
    segments_root.mkdir(parents=True, exist_ok=True)
    merge_metadata(metadata_path, combined_df, processed_ids)

    print(f"done: {len(processed_ids)} files, {len(combined_df)} segments")
    print(f"metadata updated: {metadata_path}")


if __name__ == "__main__":
    main()

