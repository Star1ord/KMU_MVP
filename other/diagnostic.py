from pathlib import Path
import os

paths = [
    'models/nlp/model.pkl',
    'models/cv_audio/model.joblib',
]

print("Model files check:")
for path in paths:
    p = Path(path)
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    print(f"  {'✓' if exists else '✗'} {path} ({size} bytes)")

print("\nDirectory structure:")
for root, dirs, files in os.walk('models'):
    level = root.replace('models', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')