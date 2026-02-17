"""Example: run inference on the provided sample video

Commands:
  # Video-only fast test
  python -c "from inference.video_ensemble import process_video_for_prediction, predict_with_video_model; csv,err = process_video_for_prediction('examples/sample_data/215_2025.12.22.mp4'); print(csv, err); print(predict_with_video_model(csv))"

  # Full (audio+NLP+video) — slower
  python src/preprocessing/analyze_new_media.py examples/sample_data/215_2025.12.22.mp4 --export examples/out.json

Or run the convenience script:
  python examples/run_sample_inference.py
"""
