"""Diagnostic: inspect feature matching between two frames.

When a mosaic won't form, this answers *why* by looking at a single pair:

* **few matches at all** -> the frames barely overlap (capture spacing too
  large); no feature stitcher can help.
* **many matches, few RANSAC inliers** -> the correspondences are inconsistent
  (repetitive texture, parallax, or residual lens distortion); fixable in code
  or with different settings.

It prints the match breakdown and saves a side-by-side visualisation so you can
eyeball whether the green correspondence lines look like one consistent shift
(real overlap) or random spaghetti (false matches).

Run on a folder/glob to inspect its first two ordered frames, or pass two files:

    transect-stitch-inspect ./frames --detector sift --undistort -0.3
    transect-stitch-inspect a.jpg b.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .stitch import (
    StitchConfig,
    _build_detector,
    _build_matcher,
    _detect_gray,
    _good_matches,
    _load_image,
)


def _resolve_pair(inputs: Sequence[str]):
    """Return two image paths: explicit pair, or first two ordered frames."""
    from .metadata import load_frame_infos, order_frames

    if len(inputs) == 1:
        frames = order_frames(load_frame_infos(inputs))
        if len(frames) < 2:
            raise SystemExit(
                f"error: need at least two images to inspect; found {len(frames)} "
                f"in {inputs[0]!r}."
            )
        return frames[0].path, frames[1].path
    return Path(inputs[0]), Path(inputs[1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="transect-stitch-inspect",
        description="Report feature matches between two frames and save a "
        "visualisation, to diagnose why a pair won't register.",
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Two image files, or a folder/glob (its first two ordered frames are used).",
    )
    p.add_argument("--detector", choices=["orb", "sift"], default="orb")
    p.add_argument("--undistort", type=float, default=0.0, metavar="K1")
    p.add_argument("--no-clahe", dest="clahe", action="store_false")
    p.add_argument("--max-features", type=int, default=4000)
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--ransac-thresh", type=float, default=4.0)
    p.add_argument(
        "--max-dim",
        type=int,
        default=1600,
        help="Downscale frames to this longest side for a manageable visualisation "
        "(default: 1600). Overlap fraction is unaffected.",
    )
    p.add_argument("-o", "--output", default="inspect_matches.jpg", help="Visualisation path.")
    args = p.parse_args(argv)

    import cv2
    import numpy as np

    cfg = StitchConfig(
        detector=args.detector,
        max_features=args.max_features,
        ratio=args.ratio,
        ransac_thresh=args.ransac_thresh,
        undistort=args.undistort,
        clahe=args.clahe,
        max_dim=args.max_dim,
    )

    path1, path2 = _resolve_pair(args.inputs)
    print(f"frame A: {path1.name}")
    print(f"frame B: {path2.name}")

    detector = _build_detector(cfg)
    matcher = _build_matcher(cfg)
    img1 = _load_image(path1, cfg)
    img2 = _load_image(path2, cfg)
    kp1, desc1 = detector.detectAndCompute(_detect_gray(img1, cfg), None)
    kp2, desc2 = detector.detectAndCompute(_detect_gray(img2, cfg), None)
    print(f"keypoints: {len(kp1)} (A) / {len(kp2)} (B)")

    good = _good_matches(matcher, desc1, desc2, cfg.ratio)
    print(f"ratio-good matches: {len(good)}")

    n_inliers = 0
    if len(good) >= 3:
        src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh
        )
        if inliers is not None:
            n_inliers = int(inliers.sum())
        print(f"RANSAC inliers: {n_inliers} ({100 * n_inliers / max(1, len(good)):.0f}% of matches)")
        if matrix is not None and n_inliers > 0:
            scale = float(np.sqrt(abs(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])))
            angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
            tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
            print(f"estimate: scale={scale:.3f}  rotation={angle:.1f}deg  "
                  f"shift=({tx:.0f},{ty:.0f})px of a {img1.shape[1]}px-wide frame")

    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if not cv2.imwrite(args.output, vis):
        print(f"error: could not write {args.output}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}  (green lines connect matched features A<->B)")

    # --- plain-language verdict ---
    if len(good) < 12:
        print("\nVERDICT: too few matches -> the frames barely overlap. The capture "
              "interval is likely too coarse for feature stitching.")
    elif n_inliers < 12:
        print("\nVERDICT: plenty of matches but few are geometrically consistent. If the "
              "green lines look like random spaghetti, it's false matches (repetitive "
              "texture/parallax). If they look parallel, loosening --ransac-thresh may help.")
    else:
        print("\nVERDICT: this pair registers fine with these settings.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
