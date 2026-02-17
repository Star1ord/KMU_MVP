"""Run sample video through video-only and full pipeline tests.

Usage:
  python examples/run_sample_inference.py

This script uses the sample video at `examples/sample_data/215_2025.12.22.mp4`.
"""
import sys
from pathlib import Path

# Ensure repository root is on PYTHONPATH so imports like `inference` and `src` work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIDEO = Path(__file__).parent / 'sample_data' / '215_2025.12.22.mp4'

print(f"Using sample video: {VIDEO}")

# Video-only test
print('\n=== VIDEO-ONLY TEST ===')
try:
    from inference.video_ensemble import process_video_for_prediction, predict_with_video_model
    csv_path, err = process_video_for_prediction(str(VIDEO))
    print('CSV path:', csv_path)
    print('Error:', err)
    if csv_path:
        pred = predict_with_video_model(csv_path)
        print('Video model result:', pred)
except Exception as e:
    print('Video-only test failed:', e)

# Full pipeline test (audio + NLP + video)
print('\n=== FULL PIPELINE TEST (this may take a while) ===')
try:
    # Import analyze_new_media function from src package
    from src.preprocessing.analyze_new_media import analyze_new_media_file

    result = analyze_new_media_file(str(VIDEO), label=None, add_to_training=False, skip_transcription=True)
    print('Full pipeline result saved to out.json (if export used).')
    print('Result summary:')
    print('  success:', result.get('success'))
    print('  prediction:', result.get('prediction'))
    print('  risk_score:', result.get('risk_score'))
    if result.get('video_result'):
        print('  video_result:', result.get('video_result'))
except Exception as e:
    print('Full pipeline test failed:', e)
