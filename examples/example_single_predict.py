"""Example commands for running inference on your own video file.

Commands:
  # Video-only fast test
  python -c "from inference.video_ensemble import process_video_for_prediction, predict_with_video_model; csv, err = process_video_for_prediction(r'path\\to\\video.mp4'); print(csv, err); print(predict_with_video_model(csv))"

  # Full audio + NLP + video path
  python examples/run_sample_inference.py path\to\video.mp4
"""
