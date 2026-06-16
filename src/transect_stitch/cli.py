"""Command-line interface for Transect Stitch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .metadata import (
    ORDER_AUTO,
    ORDER_FILENAME,
    ORDER_GPS,
    ORDER_TIME,
    gps_gaps_m,
    load_frame_infos,
    order_frames,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transect-stitch",
        description="Stitch USV transect imagery (time-lapse or individual "
        "photos with GPS timestamps) into one long mosaic.",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Image folder(s), glob(s) (e.g. 'frames/*.jpg'), or individual files.",
    )
    p.add_argument("-o", "--output", help="Output mosaic path (required unless --dry-run).")
    p.add_argument(
        "--order",
        choices=[ORDER_AUTO, ORDER_GPS, ORDER_TIME, ORDER_FILENAME],
        default=ORDER_AUTO,
        help="Frame ordering strategy (default: auto).",
    )
    p.add_argument(
        "--detector",
        choices=["orb", "sift"],
        default="orb",
        help="Feature detector (default: orb). SIFT is more robust but slower.",
    )
    p.add_argument(
        "--blend",
        choices=["feather", "overwrite"],
        default="feather",
        help="Overlap blending (default: feather).",
    )
    p.add_argument(
        "--max-dim",
        type=int,
        default=0,
        help="Downscale each frame so its longest side <= this many px "
        "(0 = full resolution). Speeds up large jobs.",
    )
    p.add_argument(
        "--max-features", type=int, default=4000, help="Max features per frame (default: 4000)."
    )
    p.add_argument(
        "--min-matches",
        type=int,
        default=12,
        help="Minimum good matches to accept a pair (default: 12).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report discovered/ordered frames and GPS track without stitching.",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _print_dry_run(frames) -> None:
    gaps = gps_gaps_m(frames)
    print(f"Discovered {len(frames)} frame(s) in transect order:\n")
    header = f"{'#':>4}  {'name':<28} {'time (src)':<22} {'lat,lon':<22} {'gap(m)':>8}"
    print(header)
    print("-" * len(header))
    for i, (f, gap) in enumerate(zip(frames, gaps)):
        ts = f.timestamp.isoformat() if f.timestamp else "-"
        ts = f"{ts} ({f.timestamp_source})"
        loc = f"{f.lat:.5f},{f.lon:.5f}" if f.has_gps else "-"
        gap_s = f"{gap:8.1f}" if gap is not None else "       -"
        print(f"{i:>4}  {f.path.name[:28]:<28} {ts:<22} {loc:<22} {gap_s}")

    valid_gaps = [g for g in gaps if g is not None]
    if valid_gaps:
        total = sum(valid_gaps)
        biggest = max(valid_gaps)
        print(
            f"\nGPS track: {len(valid_gaps)} segment(s), "
            f"~{total:.1f} m total, largest gap {biggest:.1f} m."
        )
        if biggest > 5 * (total / len(valid_gaps)):
            print("  warning: a gap is much larger than average — possible dropout "
                  "or out-of-order frames.")
    else:
        print("\nNo GPS coordinates found; ordering relied on timestamps/filenames.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    frames = load_frame_infos(args.inputs)
    if not frames:
        print("error: no images found for the given inputs.", file=sys.stderr)
        return 2

    try:
        ordered = order_frames(frames, args.order)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        _print_dry_run(ordered)
        return 0

    if not args.output:
        print("error: -o/--output is required unless --dry-run is set.", file=sys.stderr)
        return 2

    # Import the heavy path only once we know we're actually stitching.
    try:
        from .stitch import StitchConfig, StitchError, stitch_frames
    except ImportError as exc:  # pragma: no cover
        print(f"error: imaging dependencies missing ({exc}). Run 'pip install -e .'.",
              file=sys.stderr)
        return 3

    cfg = StitchConfig(
        detector=args.detector,
        max_features=args.max_features,
        min_matches=args.min_matches,
        blend=args.blend,
        max_dim=args.max_dim,
    )

    def progress(i, total, msg):
        if not args.quiet:
            print(f"[{i + 1}/{total}] {msg}", file=sys.stderr)

    try:
        mosaic = stitch_frames(ordered, cfg, progress=progress)
    except StitchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    import cv2

    out_path = Path(args.output)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), mosaic):
        print(f"error: failed to write {out_path}", file=sys.stderr)
        return 1

    h, w = mosaic.shape[:2]
    print(f"Wrote {out_path} ({w}x{h} px from {len(ordered)} frames).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
