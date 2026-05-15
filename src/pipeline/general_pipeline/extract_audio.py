import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import shutil

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline.media_utils import MediaSample, discover_media, filter_samples
from src.utils.paths import AUDIO_WAV_DIR


def convert_video_to_wav(
    sample: MediaSample,
    sample_rate: int,
    normalize_db: float,
) -> bool:
    if sample.video_source_path is None:
        return True

    output_path = sample.wav_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        str(sample.video_source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-af",
        f"loudnorm=I={normalize_db}:TP=-1.5:LRA=11",
        "-y",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not output_path.exists() or output_path.stat().st_size <= 0:
            print(f"error: ffmpeg completed but wav was not created for {sample.file_id}")
            return False
        print(f"wav created: {output_path}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"error converting {sample.video_source_path.name}: {exc.stderr}")
        return False


def prepare_audio(
    input_dir: Path,
    sample_rate: int,
    normalize_db: float,
    file_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> None:
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg is not installed or not available in PATH")
        return

    samples = discover_media(input_dir)
    if not samples:
        print(f"warning: no label folders 0/1 found in {input_dir}")
        return

    target_samples = filter_samples(samples, file_ids)
    video_samples = [s for s in target_samples if s.video_source_path]

    if not video_samples:
        print("no video files to convert, skipping")
        return

    tasks = [s for s in video_samples if s.needs_conversion(force)]
    if limit is not None:
        tasks = tasks[:limit]

    if not tasks:
        print("all videos already converted to wav, using existing audio")
        return

    print(f"converting {len(tasks)} of {len(video_samples)} available videos...")

    success = 0
    for sample in tasks:
        if convert_video_to_wav(sample, sample_rate, normalize_db):
            success += 1

    print(f"done: {success}/{len(tasks)} videos converted to wav")


def main():
    parser = argparse.ArgumentParser(description="prepare audio: convert video from data/audio_wav/<label>/")
    parser.add_argument("--input-dir", type=str, default=str(AUDIO_WAV_DIR))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--normalize-db", type=float, default=-20.0)
    parser.add_argument("--file-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    prepare_audio(
        input_dir=Path(args.input_dir),
        sample_rate=args.sample_rate,
        normalize_db=args.normalize_db,
        file_ids=args.file_ids,
        limit=args.limit,
        force=args.force,
    )


if __name__ == "__main__":
    main()
