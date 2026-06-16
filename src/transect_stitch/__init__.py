"""Transect Stitch — stitch USV transect imagery into a single long mosaic."""

from .metadata import FrameInfo, load_frame_infos, order_frames
from .stitch import StitchConfig, stitch_frames

__version__ = "0.1.0"

__all__ = [
    "FrameInfo",
    "load_frame_infos",
    "order_frames",
    "StitchConfig",
    "stitch_frames",
    "__version__",
]
