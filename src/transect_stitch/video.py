"""Video frame extraction for transect stitching.

GoPro and other action cameras produce video with enough inter-frame overlap for
feature matching. This module extracts a strided subset of frames and presents
them as FrameInfo objects so the normal stitching pipeline can process them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from .metadata import FrameInfo

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts", ".m2ts", ".m4v", ".3gp"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def extract_frames(
    video_path: Path,
    out_dir: Path,
    stride: int = 5,
    max_dim: int = 0,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> list[FrameInfo]:
    """Extract every ``stride``-th frame from ``video_path`` into ``out_dir``.

    Returns FrameInfo objects with synthetic timestamps from the video fps, so
    the stitcher receives them in the correct capture order.

    A stride of 5 on 30 fps footage gives ~6 frames/sec, which usually provides
    good overlap for slow-moving USV / snorkel surveys.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ts = datetime(1970, 1, 1, tzinfo=timezone.utc)
    infos: list[FrameInfo] = []
    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            if max_dim > 0:
                h, w = frame.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(
                        frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                    )
            p = out_dir / f"frame_{written:06d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            ts = base_ts + timedelta(seconds=frame_idx / fps)
            infos.append(FrameInfo(path=p, timestamp=ts, timestamp_source="video"))
            written += 1
            if progress:
                pct = int(100 * frame_idx / max(1, total))
                progress(frame_idx, total, f"extracting frame {written} ({pct}%)")
        frame_idx += 1

    cap.release()
    return infos
