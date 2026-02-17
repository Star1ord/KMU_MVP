from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

AUDIO_EXTENSIONS = {".wav"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


@dataclass
class MediaSample:

    file_id: str
    label: int
    label_dir: Path
    wav_path: Path
    audio_source_path: Optional[Path] = None
    video_source_path: Optional[Path] = None

    def audio_subpath(self) -> Path:
        return Path(str(self.label)) / f"{self.file_id}.wav"

    def transcript_path(self, transcripts_root: Path) -> Path:
        return transcripts_root / str(self.label) / f"{self.file_id}.json"

    def legacy_transcript_path(self, transcripts_root: Path) -> Path:
        return transcripts_root / f"{self.file_id}.json"

    def segments_dir(self, segments_root: Path) -> Path:
        return segments_root / self.file_id

    def transcript_exists(self, transcripts_root: Path) -> bool:
        return (
            self.transcript_path(transcripts_root).exists()
            or self.legacy_transcript_path(transcripts_root).exists()
        )

    def needs_transcript(self, transcripts_root: Path, force: bool = False) -> bool:
        return force or not self.transcript_exists(transcripts_root)

    def needs_conversion(self, force: bool = False) -> bool:
        if self.video_source_path is None:
            return False
        if force:
            return True
        if not self.wav_path.exists():
            return True
        try:
            return self.video_source_path.stat().st_mtime > self.wav_path.stat().st_mtime
        except FileNotFoundError:
            return True


def _iter_label_dirs(audio_root: Path) -> Iterable[Path]:
    if not audio_root.exists():
        return []
    for candidate in sorted(p for p in audio_root.iterdir() if p.is_dir()):
        try:
            label = int(candidate.name)
        except ValueError:
            continue
        if label in {0, 1}:
            yield candidate


def discover_media(audio_root: Path, transcripts_root: Optional[Path] = None) -> List[MediaSample]:
    samples: Dict[tuple[int, str], MediaSample] = {}
    for label_dir in _iter_label_dirs(audio_root):
        label = int(label_dir.name)
        for media_file in sorted(p for p in label_dir.iterdir() if p.is_file()):
            ext = media_file.suffix.lower()
            if ext not in AUDIO_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
                continue

            file_id = media_file.stem
            key = (label, file_id)
            if key not in samples:
                wav_path = label_dir / f"{file_id}.wav"
                samples[key] = MediaSample(
                    file_id=file_id,
                    label=label,
                    label_dir=label_dir,
                    wav_path=wav_path,
                )

            sample = samples[key]
            if ext in AUDIO_EXTENSIONS:
                sample.wav_path = media_file
                sample.audio_source_path = media_file
            else:
                sample.video_source_path = media_file

    by_id: Dict[str, MediaSample] = {}
    conflicts: List[tuple[str, MediaSample, MediaSample]] = []
    
    for sample in samples.values():
        existing = by_id.get(sample.file_id)
        if existing and existing.label != sample.label:
            conflicts.append((sample.file_id, existing, sample))
        else:
            by_id[sample.file_id] = sample

    for file_id, existing_sample, new_sample in conflicts:
        resolved_sample = None
        
        if transcripts_root and transcripts_root.exists():
            existing_has_transcript = existing_sample.transcript_exists(transcripts_root)
            new_has_transcript = new_sample.transcript_exists(transcripts_root)
            
            if existing_has_transcript and not new_has_transcript:
                resolved_sample = existing_sample
                print(
                    f"warning: file_id '{file_id}' found in both label {existing_sample.label} and {new_sample.label}. "
                    f"keeping label {existing_sample.label} (transcript exists)"
                )
            elif new_has_transcript and not existing_has_transcript:
                resolved_sample = new_sample
                print(
                    f"warning: file_id '{file_id}' found in both label {existing_sample.label} and {new_sample.label}. "
                    f"keeping label {new_sample.label} (transcript exists)"
                )
        
        if resolved_sample is None:
            if existing_sample.label > new_sample.label:
                resolved_sample = existing_sample
                print(
                    f"warning: file_id '{file_id}' found in both label {existing_sample.label} and {new_sample.label}. "
                    f"keeping label {existing_sample.label} (higher priority)"
                )
            else:
                resolved_sample = new_sample
                print(
                    f"warning: file_id '{file_id}' found in both label {existing_sample.label} and {new_sample.label}. "
                    f"keeping label {new_sample.label} (higher priority)"
                )
        
        by_id[file_id] = resolved_sample

    return sorted(by_id.values(), key=lambda s: (s.label, s.file_id))


def build_index(samples: Sequence[MediaSample]) -> Dict[str, MediaSample]:
    return {sample.file_id: sample for sample in samples}


def filter_samples(
    samples: Sequence[MediaSample],
    file_ids: Optional[Sequence[str]] = None,
) -> List[MediaSample]:
    if not file_ids:
        return list(samples)

    allowed = {fid.strip() for fid in file_ids if fid.strip()}
    if not allowed:
        return list(samples)

    result = [sample for sample in samples if sample.file_id in allowed]
    missing = allowed.difference({sample.file_id for sample in result})
    if missing:
        print(f"warning: some file_ids not found in audio_wav: {', '.join(sorted(missing))}")
    return result


def map_file_labels(audio_root: Path) -> Dict[str, int]:
    return {sample.file_id: sample.label for sample in discover_media(audio_root)}


def migrate_transcript(sample: MediaSample, transcripts_root: Path) -> bool:
    legacy_path = sample.legacy_transcript_path(transcripts_root)
    new_path = sample.transcript_path(transcripts_root)

    if not legacy_path.exists() or new_path.exists():
        return False

    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_path, new_path)
    return True



