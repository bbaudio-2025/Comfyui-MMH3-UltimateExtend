"""Package holding the MMH3 Spatial Extend Video node implementations.

`extend_video.py` is the Extend Video / Tile Plan / Overlap Fade Override nodes
and `helpers.py` the shared H3 helpers they depend on. This module re-exports the
node classes so the plugin's root `__init__.py` can import them from `.nodes`.
"""

from .extend_video import (MMH3SpatialExtendVideo, MMH3MaskPreview,
                           MMH3LastQuadrantPatch)
from .tile_editor import MMH3SpatialTileEditor
from .mmh3_preview import MMH3LatentPreview
from .temporal_extend import (MMH3SampleParams, MMH3TemporalExtendVideo,
                              MMH3TemporalOverlapParams,
                              MMH3TemporalOverlapSimple)
from .temporal_tile_editor import MMH3TemporalTileEditor

# explicit re-exports: they exist for the root __init__.py, and declaring them
# keeps the static checker honest (any remaining diagnostic is a real one)
__all__ = [
    "MMH3SpatialExtendVideo", "MMH3MaskPreview", "MMH3LastQuadrantPatch",
    "MMH3SpatialTileEditor", "MMH3LatentPreview", "MMH3SampleParams",
    "MMH3TemporalExtendVideo", "MMH3TemporalOverlapParams",
    "MMH3TemporalOverlapSimple", "MMH3TemporalTileEditor",
]
