# Proxy module to maintain backward compatibility.
# The actual media utils live under `pipeline.general_pipeline.media_utils`.
from pipeline.general_pipeline.media_utils import (
    MediaSample,
    discover_media,
    filter_samples,
    build_index,
    map_file_labels,
    migrate_transcript,
)

__all__ = [
    "MediaSample",
    "discover_media",
    "filter_samples",
    "build_index",
    "map_file_labels",
    "migrate_transcript",
]
