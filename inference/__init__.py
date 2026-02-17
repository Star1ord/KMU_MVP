"""Top-level inference package proxy.

This package exists to make imports like "from inference.video_ensemble import ..."
work when running from the repository root. The real implementation lives in
`src/inference/` and is loaded by the proxy module(s) below.
"""

__all__ = ["video_ensemble"]
