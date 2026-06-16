"""Image discovery, EXIF metadata extraction, and frame ordering.

Frames are ordered by the most reliable signal available:

1. GPS timestamp  (``GPSDateStamp`` + ``GPSTimeStamp``, UTC)
2. EXIF capture time  (``DateTimeOriginal``)
3. Natural filename sort  (so ``frame_2`` precedes ``frame_10``)

GPS coordinates, when present, are extracted too so a dataset can be inspected
(track + gap detection) before committing to a long stitch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .geo import haversine_m

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Order strategies understood by :func:`order_frames`.
ORDER_AUTO = "auto"
ORDER_GPS = "gps"
ORDER_TIME = "time"
ORDER_FILENAME = "filename"


@dataclass
class FrameInfo:
    """Everything we know about one image before it is read for stitching."""

    path: Path
    timestamp: Optional[datetime] = None  # best available capture time (UTC if from GPS)
    lat: Optional[float] = None
    lon: Optional[float] = None
    timestamp_source: str = "none"  # "gps" | "exif" | "none"

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None


_NATURAL_RE = re.compile(r"(\d+)")


def _natural_key(path: Path):
    """Split a name into text/number chunks for human-friendly sorting."""
    return [int(part) if part.isdigit() else part.lower() for part in _NATURAL_RE.split(path.name)]


def discover_images(inputs: Sequence[str]) -> list[Path]:
    """Expand the CLI inputs (dirs, globs, explicit files) into image paths.

    Directories are scanned (non-recursively) for known image extensions.
    Strings containing glob characters are expanded. Plain paths are kept as-is.
    Results are de-duplicated while preserving first-seen order.
    """
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(c for c in p.iterdir() if c.suffix.lower() in IMAGE_EXTENSIONS)
        elif any(ch in item for ch in "*?[]"):
            # Glob relative to cwd; Path.glob needs a base, so use the parent.
            base = p.parent if str(p.parent) not in ("", ".") else Path(".")
            paths.extend(base.glob(p.name))
        elif p.exists():
            paths.append(p)
        # Silently skip non-existent plain paths; the caller reports an empty set.

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen and p.suffix.lower() in IMAGE_EXTENSIONS:
            seen.add(rp)
            unique.append(p)
    return unique


def _to_float(value) -> Optional[float]:
    """Coerce an EXIF rational/tuple/number to float."""
    try:
        if isinstance(value, tuple) and len(value) == 2:  # (num, den) rational
            return float(value[0]) / float(value[1]) if value[1] else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_deg(dms, ref) -> Optional[float]:
    """Convert EXIF degrees/minutes/seconds + hemisphere ref to signed decimal."""
    try:
        d = _to_float(dms[0]) or 0.0
        m = _to_float(dms[1]) or 0.0
        s = _to_float(dms[2]) or 0.0
    except (TypeError, IndexError):
        return None
    deg = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        deg = -deg
    return deg


def _parse_gps_datetime(gps_ifd) -> Optional[datetime]:
    """Build a UTC datetime from GPSDateStamp (1:2:3) + GPSTimeStamp (h,m,s)."""
    date_stamp = gps_ifd.get(29)  # GPSDateStamp, "YYYY:MM:DD"
    time_stamp = gps_ifd.get(7)  # GPSTimeStamp, (h, m, s) rationals
    if not date_stamp or not time_stamp:
        return None
    try:
        y, mo, d = (int(x) for x in str(date_stamp).split(":"))
        h = int(_to_float(time_stamp[0]) or 0)
        mi = int(_to_float(time_stamp[1]) or 0)
        sec = int(_to_float(time_stamp[2]) or 0)
        return datetime(y, mo, d, h, mi, sec, tzinfo=timezone.utc)
    except (ValueError, TypeError, IndexError):
        return None


def _parse_exif_datetime(value: str) -> Optional[datetime]:
    """Parse an EXIF ``DateTimeOriginal`` string ("YYYY:MM:DD HH:MM:SS")."""
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def read_frame_info(path: Path) -> FrameInfo:
    """Read EXIF for a single image, degrading gracefully when tags are absent."""
    info = FrameInfo(path=path)
    try:
        from PIL import Image  # imported lazily so non-imaging code stays dep-free
    except ImportError:
        return info  # no Pillow -> filename ordering only

    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except Exception:
        return info

    if not exif:
        return info

    # --- GPS block (IFD 0x8825) ---
    try:
        gps_ifd = exif.get_ifd(0x8825)
    except Exception:
        gps_ifd = {}
    if gps_ifd:
        lat = _dms_to_deg(gps_ifd.get(2), gps_ifd.get(1))  # GPSLatitude / Ref
        lon = _dms_to_deg(gps_ifd.get(4), gps_ifd.get(3))  # GPSLongitude / Ref
        if lat is not None and lon is not None:
            info.lat, info.lon = lat, lon
        gps_dt = _parse_gps_datetime(gps_ifd)
        if gps_dt is not None:
            info.timestamp = gps_dt
            info.timestamp_source = "gps"

    # --- Fall back to capture time if no GPS time ---
    if info.timestamp is None:
        # 0x9003 = DateTimeOriginal, 0x0132 = DateTime (last modified)
        raw = exif.get(0x9003) or exif.get(0x0132)
        if raw:
            dt = _parse_exif_datetime(raw)
            if dt is not None:
                info.timestamp = dt
                info.timestamp_source = "exif"

    return info


def load_frame_infos(inputs: Sequence[str]) -> list[FrameInfo]:
    """Discover images and read metadata for each."""
    return [read_frame_info(p) for p in discover_images(inputs)]


def order_frames(frames: Iterable[FrameInfo], strategy: str = ORDER_AUTO) -> list[FrameInfo]:
    """Return frames sorted according to ``strategy``.

    ``auto`` uses GPS/EXIF timestamps when *every* frame has one, otherwise
    falls back to a natural filename sort (mixing the two is unreliable).
    """
    frames = list(frames)
    if not frames:
        return frames

    def by_name(fs):
        return sorted(fs, key=lambda f: _natural_key(f.path))

    if strategy == ORDER_FILENAME:
        return by_name(frames)

    if strategy in (ORDER_GPS, ORDER_TIME, ORDER_AUTO):
        if strategy == ORDER_GPS:
            usable = all(f.timestamp_source == "gps" for f in frames)
        elif strategy == ORDER_TIME:
            usable = all(f.timestamp is not None for f in frames)
        else:  # auto
            usable = all(f.timestamp is not None for f in frames)
        if usable:
            return sorted(frames, key=lambda f: f.timestamp)  # type: ignore[arg-type]
        if strategy == ORDER_AUTO:
            return by_name(frames)
        raise ValueError(
            f"order='{strategy}' requested but not all frames have the required "
            f"metadata. Use --order filename or fix the source images."
        )

    raise ValueError(f"Unknown order strategy: {strategy!r}")


def gps_gaps_m(frames: Sequence[FrameInfo]) -> list[Optional[float]]:
    """Distance (m) between each frame and the previous one with GPS.

    Element 0 is always ``None``; entries are ``None`` where either frame lacks
    coordinates. Useful for flagging dropouts / out-of-order data.
    """
    gaps: list[Optional[float]] = [None]
    for prev, cur in zip(frames, frames[1:]):
        if prev.has_gps and cur.has_gps:
            gaps.append(haversine_m(prev.lat, prev.lon, cur.lat, cur.lon))  # type: ignore[arg-type]
        else:
            gaps.append(None)
    return gaps
