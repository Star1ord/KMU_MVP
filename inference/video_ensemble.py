"""Proxy loader for `inference.video_ensemble`.

Loads the real implementation from `src/inference/video_ensemble.py` so that
imports from the repository root work without modifying PYTHONPATH.
"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
SRC_MODULE_PATH = ROOT / 'src' / 'inference' / 'video_ensemble.py'

if not SRC_MODULE_PATH.exists():
    raise ImportError(f"Source module not found: {SRC_MODULE_PATH}")

spec = importlib.util.spec_from_file_location(__name__ + "._src", str(SRC_MODULE_PATH))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Re-export public attributes from the source module
for attr in dir(module):
    if not attr.startswith("_"):
        globals()[attr] = getattr(module, attr)

__all__ = [k for k in globals().keys() if not k.startswith("_")]
