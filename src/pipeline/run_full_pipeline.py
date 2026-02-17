"""Proxy to the general pipeline runner.
This ensures scripts that call "src/pipeline/run_full_pipeline.py" work.
"""
import sys
from pathlib import Path
import runpy

# Ensure `src` package is importable when this script is invoked directly
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == '__main__':
    # Execute the general pipeline script as a script so it behaves the same
    target = Path(__file__).resolve().parent / 'general_pipeline' / 'run_full_pipeline.py'
    runpy.run_path(str(target), run_name='__main__')
