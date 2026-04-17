"""Run a local video through the video-only and full-pipeline flows.

Usage:
  python examples/run_sample_inference.py path\to\video.mp4

If no path is provided, the script will try:
  1. TEST_VIDEO_PATH environment variable
  2. the first video found in examples/sample_data/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def resolve_video_path(raw_path: str | None) -> Path | None:
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
        return path if path.exists() else None

    env_path = os.getenv("TEST_VIDEO_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path

    sample_dir = ROOT / "examples" / "sample_data"
    if sample_dir.exists():
        for pattern in ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"):
            matches = sorted(sample_dir.glob(pattern))
            if matches:
                return matches[0].resolve()

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local inference checks for a video file.")
    parser.add_argument("video", nargs="?", help="Path to a local video file")
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    if video_path is None:
        print("No video file found.")
        print("Pass a path explicitly or set TEST_VIDEO_PATH.")
        print("Optional drop location: examples/sample_data/")
        return 1

    print(f"Using video: {video_path}")

    print("\n=== VIDEO-ONLY TEST ===")
    try:
        from inference.video_ensemble import process_video_for_prediction, predict_with_video_model

        csv_path, err = process_video_for_prediction(str(video_path))
        print("CSV path:", csv_path)
        print("Error:", err)
        if csv_path:
            pred = predict_with_video_model(csv_path)
            print("Video model result:", pred)
    except Exception as exc:
        print("Video-only test failed:", exc)

    print("\n=== FULL PIPELINE TEST (this may take a while) ===")
    try:
        from src.preprocessing.analyze_new_media import analyze_new_media_file

        result = analyze_new_media_file(
            str(video_path),
            label=None,
            add_to_training=False,
            skip_transcription=True,
        )
        payload = result.get("formatted_result", result) if isinstance(result, dict) else {}
        print("Result summary:")
        print("  success:", payload.get("success"))
        print("  prediction:", payload.get("prediction"))
        print("  risk_score:", payload.get("risk_score"))
        if payload.get("video_result"):
            print("  video_result:", payload.get("video_result"))
    except Exception as exc:
        print("Full pipeline test failed:", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
