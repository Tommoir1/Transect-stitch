"""Command-line interface for Transect Stitch."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .metadata import (
    ORDER_AUTO,
    ORDER_FILENAME,
    ORDER_GPS,
    ORDER_TIME,
    chunk_frames,
    gps_gaps_m,
    load_frame_infos,
    order_frames,
    stride_frames,
)
from .video import VIDEO_EXTENSIONS, extract_frames, is_video


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transect-stitch",
        description="Stitch USV transect imagery (time-lapse or individual "
        "photos with GPS timestamps) into one long mosaic.",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Image folder(s), glob(s) (e.g. 'frames/*.jpg'), individual image files, "
        "or a video file (.mp4, .mov, .avi …).",
    )
    p.add_argument(
        "--video-stride",
        type=int,
        default=5,
        metavar="N",
        help="When the input is a video, extract every Nth frame (default: 5). "
        "At 30 fps, stride 5 = 6 frames/sec. Lower = more overlap but slower.",
    )
    p.add_argument("-o", "--output", help="Output mosaic path (required unless --dry-run).")
    p.add_argument(
        "--order",
        choices=[ORDER_AUTO, ORDER_GPS, ORDER_TIME, ORDER_FILENAME],
        default=ORDER_AUTO,
        help="Frame ordering strategy (default: auto).",
    )
    p.add_argument(
        "--preset",
        choices=["none", "underwater"],
        default="none",
        help="Apply a bundle of settings for a known-awkward capture type. "
        "'underwater' tunes for GoPro reef footage (SIFT, looser matching, "
        "lens correction, homography). Explicit flags override the preset.",
    )
    p.add_argument(
        "--detector",
        choices=["orb", "sift"],
        default="orb",
        help="Feature detector (default: orb). SIFT is more robust but slower.",
    )
    p.add_argument(
        "--transform",
        choices=["affine", "homography"],
        default="affine",
        help="Motion model per pair: affine (rigid-ish, default) or homography "
        "(planar perspective — better for flat scenes through a wide lens).",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio-test threshold (default: 0.75). Higher (e.g. 0.9) admits "
        "more matches on repetitive texture; RANSAC then filters them.",
    )
    p.add_argument(
        "--ransac-thresh",
        type=float,
        default=4.0,
        help="RANSAC reprojection tolerance in px (default: 4). Higher (e.g. 10) "
        "tolerates residual lens distortion / blur.",
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
        help="Minimum geometrically consistent (RANSAC inlier) matches to accept a "
        "pair (default: 12).",
    )
    p.add_argument(
        "--undistort",
        type=float,
        default=0.0,
        metavar="K1",
        help="Radial lens-correction strength for wide-angle/fisheye footage "
        "(e.g. -0.3 for a GoPro). 0 (default) = off. Negative values straighten "
        "barrel distortion so frames actually register.",
    )
    p.add_argument(
        "--no-clahe",
        dest="clahe",
        action="store_false",
        help="Disable CLAHE contrast enhancement before feature detection "
        "(on by default; helps low-contrast underwater frames).",
    )
    p.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.0,
        help="If >0, also require inliers/matches >= this fraction to accept a pair "
        "(rejects low-confidence registrations; default: 0 = off).",
    )
    p.add_argument(
        "--max-skip",
        type=int,
        default=5,
        help="Consecutive un-registerable frames to drop before giving up on a "
        "mosaic (default: 5). A single blurry frame won't abort the run.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame (default: 1). Thins dense time-lapses with heavy overlap.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch mode: emit one mosaic per N frames (e.g. 40). With this set, "
        "--output must be a folder. 0 (default) stitches everything into one mosaic.",
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

    # --- video input: extract frames into a temp dir, then stitch as normal ---
    video_inputs = [Path(i) for i in args.inputs if is_video(Path(i))]
    if video_inputs and len(args.inputs) > 1:
        print("error: pass a single video file, not a mix of video and images.",
              file=sys.stderr)
        return 2
    if video_inputs:
        video_path = video_inputs[0]
        tmp = Path(tempfile.mkdtemp(prefix="transect_stitch_"))
        if not args.quiet:
            print(f"Extracting every {args.video_stride}th frame from {video_path.name} …",
                  file=sys.stderr)

        def extract_progress(idx, total, msg):
            if not args.quiet:
                print(f"\r  {msg}", end="", file=sys.stderr, flush=True)

        try:
            frames = extract_frames(
                video_path, tmp, stride=args.video_stride,
                max_dim=args.max_dim, progress=extract_progress,
            )
        except RuntimeError as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
            return 2
        if not args.quiet:
            print(f"\n  extracted {len(frames)} frames to {tmp}", file=sys.stderr)
        ordered = frames  # already in capture order by timestamp
    else:
        frames = load_frame_infos(args.inputs)
        if not frames:
            print("error: no images found for the given inputs.", file=sys.stderr)
            return 2
        try:
            ordered = order_frames(frames, args.order)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.stride < 1:
        print("error: --stride must be >= 1.", file=sys.stderr)
        return 2
    ordered = stride_frames(ordered, args.stride)

    if args.dry_run:
        _print_dry_run(ordered)
        return 0

    if not args.output:
        print("error: -o/--output is required unless --dry-run is set.", file=sys.stderr)
        return 2

    # Import the heavy path only once we know we're actually stitching.
    try:
        import dataclasses

        from .stitch import StitchConfig, StitchError, apply_preset, stitch_frames
    except ImportError as exc:  # pragma: no cover
        print(f"error: imaging dependencies missing ({exc}). Run 'pip install -e .'.",
              file=sys.stderr)
        return 3

    # Start from the preset (if any), then let explicitly-passed flags override it.
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    cfg = apply_preset(StitchConfig(), args.preset)
    overrides = {}
    for dest, flags in {
        "detector": ["--detector"], "transform": ["--transform"], "ratio": ["--ratio"],
        "ransac_thresh": ["--ransac-thresh"], "max_features": ["--max-features"],
        "min_matches": ["--min-matches"], "undistort": ["--undistort"],
        "blend": ["--blend"], "max_dim": ["--max-dim"], "clahe": ["--no-clahe"],
        "min_inlier_ratio": ["--min-inlier-ratio"], "max_skip": ["--max-skip"],
    }.items():
        if any(a == f or a.startswith(f + "=") for a in argv_list for f in flags):
            overrides[dest] = getattr(args, dest)
    cfg = dataclasses.replace(cfg, **overrides)

    def progress(i, total, msg):
        if not args.quiet:
            print(f"[{i + 1}/{total}] {msg}", file=sys.stderr)

    import cv2

    # --- batch mode: one mosaic per N frames into an output folder ---
    if args.batch_size and args.batch_size > 0:
        groups = chunk_frames(ordered, args.batch_size)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = failed = 0
        for gi, group in enumerate(groups):
            if not args.quiet:
                print(f"== mosaic {gi + 1}/{len(groups)} ({len(group)} frames) ==",
                      file=sys.stderr)
            try:
                mosaic = stitch_frames(group, cfg, progress=progress)
            except StitchError as exc:
                failed += 1
                print(f"  skipped mosaic {gi + 1}: {exc}", file=sys.stderr)
                continue
            dest = out_dir / f"mosaic_{gi + 1:03d}_{group[0].path.stem}_to_{group[-1].path.stem}.jpg"
            if cv2.imwrite(str(dest), mosaic):
                written += 1
                print(f"Wrote {dest}")
            else:
                failed += 1
                print(f"  failed to write {dest}", file=sys.stderr)
        print(f"Done: {written} mosaic(s) written, {failed} skipped/failed.")
        return 0 if failed == 0 else 1

    # --- single mosaic ---
    try:
        mosaic = stitch_frames(ordered, cfg, progress=progress)
    except StitchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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
