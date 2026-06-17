"""Sequential transect mosaicking.

OpenCV's high-level ``cv2.Stitcher`` targets rotational panoramas and tends to
fail on long, mostly-linear transects. Instead we register frames pairwise and
accumulate the transforms onto one growing canvas:

    1. Detect + describe features per frame (ORB by default, SIFT optional).
    2. Match consecutive frames, filter with Lowe's ratio test.
    3. Estimate a *partial* affine (translation + rotation + uniform scale) with
       RANSAC. Limiting to partial-affine absorbs USV roll/yaw/surge without the
       runaway distortion a full homography can produce on weak overlaps.
    4. Compose each pairwise transform into a global one and warp every frame
       onto a shared canvas, blending overlaps.

cv2 / numpy are imported lazily so dataset inspection (ordering, GPS checks)
does not require the imaging stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .metadata import FrameInfo

# Optional progress callback: (index, total, message) -> None
ProgressFn = Callable[[int, int, str], None]


@dataclass
class StitchConfig:
    detector: str = "orb"  # "orb" | "sift"
    max_features: int = 4000
    ratio: float = 0.75  # Lowe ratio test threshold
    min_matches: int = 12  # min RANSAC inliers to trust a pair
    ransac_thresh: float = 4.0  # reprojection tolerance (px)
    blend: str = "feather"  # "feather" | "overwrite"
    max_dim: int = 0  # if >0, downscale each frame so max(h,w) <= max_dim

    # --- robustness for hard imagery (e.g. underwater / wide-angle action cams) ---
    undistort: float = 0.0  # radial lens-correction strength k1 (e.g. -0.3 for GoPro); 0 = off
    clahe: bool = True  # contrast-limited adaptive equalisation before feature detection
    transform: str = "affine"  # "affine" (rigid-ish) | "homography" (planar perspective)
    min_inlier_ratio: float = 0.0  # if >0, also require inliers / good-matches >= this
    max_scale: float = 4.0  # reject a pair whose estimated scale jump exceeds this (0 = off)
    max_skip: int = 5  # consecutive un-registerable frames to drop before giving up


# Presets bundle settings for common, awkward capture types so users don't have
# to hand-tune several knobs. Applied as overrides on top of the defaults.
PRESETS = {
    "underwater": dict(
        detector="sift",
        max_features=8000,
        ratio=0.9,  # repetitive algae/coral texture -> admit more matches, let RANSAC filter
        ransac_thresh=10.0,  # fisheye + motion blur -> looser geometric tolerance
        min_matches=8,
        undistort=-0.3,  # straighten GoPro barrel distortion
        clahe=True,
        transform="homography",  # flat seabed through a wide lens is a planar perspective
    ),
}


def apply_preset(cfg: "StitchConfig", name: str) -> "StitchConfig":
    """Return a copy of ``cfg`` with the named preset's overrides applied."""
    if not name or name == "none":
        return cfg
    if name not in PRESETS:
        raise StitchError(f"Unknown preset: {name!r} (known: {', '.join(PRESETS)}).")
    import dataclasses

    return dataclasses.replace(cfg, **PRESETS[name])


class StitchError(RuntimeError):
    """Raised when frames cannot be registered into a mosaic."""


def _build_detector(cfg: StitchConfig):
    import cv2

    if cfg.detector == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise StitchError(
                "SIFT is unavailable in this OpenCV build; use --detector orb."
            )
        return cv2.SIFT_create(nfeatures=cfg.max_features)
    if cfg.detector == "orb":
        return cv2.ORB_create(nfeatures=cfg.max_features)
    raise StitchError(f"Unknown detector: {cfg.detector!r}")


def _build_matcher(cfg: StitchConfig):
    import cv2

    # ORB -> binary descriptors (Hamming); SIFT -> float descriptors (L2).
    norm = cv2.NORM_HAMMING if cfg.detector == "orb" else cv2.NORM_L2
    return cv2.BFMatcher(norm, crossCheck=False)


def _undistort(img, k1: float, k2: float = 0.0):
    """Approximate radial lens correction (negative k1 straightens barrel/fisheye).

    No calibration is available, so we assume a centred principal point and a
    focal length of the longest image side. This is rough but enough to stop the
    edges of a wide-angle frame curving — which is what breaks the affine model
    on action-cam footage. The central overlap region (where most matches live)
    is straightened, letting RANSAC find far more consistent inliers.
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    f = float(max(w, h))
    k = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    d = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
    return cv2.undistort(img, k, d)


def _load_image(path: Path, cfg: StitchConfig):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise StitchError(f"Could not read image: {path}")
    if cfg.undistort != 0.0:
        img = _undistort(img, cfg.undistort)
    if cfg.max_dim > 0:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > cfg.max_dim:
            scale = cfg.max_dim / longest
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _detect_gray(img, cfg: StitchConfig):
    """Grayscale used for feature detection, optionally CLAHE-enhanced.

    Underwater / low-contrast frames give up far more features once local
    contrast is equalised, so this is on by default.
    """
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cfg.clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def _good_matches(matcher, desc1, desc2, ratio: float):
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _estimate_pairwise(kp1, kp2, matches, cfg: StitchConfig):
    """Estimate a 3x3 transform mapping frame2 -> frame1.

    Uses a partial-affine (rigid-ish) model by default, or a full homography
    when ``cfg.transform == 'homography'`` (better for a flat scene viewed
    through a wide/fisheye lens). Returns ``(matrix3x3_or_None, n_inliers)``;
    ``n_inliers`` is reported even on rejection so callers can explain *why* a
    pair failed (too few geometrically consistent matches, not just raw matches).
    """
    import cv2
    import numpy as np

    if len(matches) < cfg.min_matches:
        return None, 0
    src = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    if cfg.transform == "homography":
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_thresh)
    else:
        affine, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh
        )
        matrix = _to_3x3(affine) if affine is not None else None
    if matrix is None or inliers is None:
        return None, 0

    n_inliers = int(inliers.sum())
    if n_inliers < cfg.min_matches:
        return None, n_inliers
    if cfg.min_inlier_ratio > 0 and n_inliers < cfg.min_inlier_ratio * len(matches):
        return None, n_inliers
    # Guard against degenerate fits that imply an implausible zoom between
    # consecutive transect frames (scale should be ~1).
    scale = float(np.sqrt(abs(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])))
    if cfg.max_scale > 0 and (scale > cfg.max_scale or scale < 1.0 / cfg.max_scale):
        return None, n_inliers
    return matrix, n_inliers


def _to_3x3(affine):
    import numpy as np

    h = np.eye(3, dtype=np.float64)
    h[:2, :] = affine
    return h


def stitch_frames(
    frames: Sequence[FrameInfo],
    config: Optional[StitchConfig] = None,
    progress: Optional[ProgressFn] = None,
):
    """Stitch ordered ``frames`` into a single mosaic (returns a BGR ndarray).

    Frames must already be in transect order (see ``metadata.order_frames``).
    """
    import cv2
    import numpy as np

    cfg = config or StitchConfig()
    if len(frames) == 0:
        raise StitchError("No frames to stitch.")

    detector = _build_detector(cfg)
    matcher = _build_matcher(cfg)
    total = len(frames)

    def report(i, msg):
        if progress:
            progress(i, total, msg)

    # First frame anchors the global coordinate system at identity.
    report(0, f"loading {frames[0].path.name}")
    base = _load_image(frames[0].path, cfg)
    if total == 1:
        return base
    anchor_kp, anchor_desc = detector.detectAndCompute(_detect_gray(base, cfg), None)
    anchor_name = frames[0].path.name

    global_h = [np.eye(3, dtype=np.float64)]  # placed frame i -> global
    images = [base]
    cur_global = np.eye(3, dtype=np.float64)  # global transform of the current anchor

    skipped = 0
    consecutive_skips = 0
    for i in range(1, total):
        report(i, f"registering {frames[i].path.name}")
        img = _load_image(frames[i].path, cfg)
        kp, desc = detector.detectAndCompute(_detect_gray(img, cfg), None)

        # Match against the last *successfully placed* frame, so a single bad
        # (blurry / textureless) frame is bridged over rather than aborting.
        matches = _good_matches(matcher, anchor_desc, desc, cfg.ratio)
        pair_h, n_inliers = _estimate_pairwise(anchor_kp, kp, matches, cfg)
        if pair_h is None:
            skipped += 1
            consecutive_skips += 1
            report(
                i,
                f"  dropped {frames[i].path.name}: {len(matches)} matches but only "
                f"{n_inliers} geometrically consistent (need >= {cfg.min_matches})",
            )
            if consecutive_skips > cfg.max_skip:
                raise StitchError(
                    f"Lost registration: {consecutive_skips} frames in a row could not "
                    f"be matched to '{anchor_name}'. The overlap or texture is too low "
                    f"(last try: {len(matches)} matches, {n_inliers} inliers). Try a "
                    f"lower --stride, --detector sift, or --undistort for wide-angle lenses."
                )
            continue

        # pair_h maps current -> anchor; compose onto the anchor's global transform.
        cur_global = cur_global @ pair_h
        global_h.append(cur_global.copy())
        images.append(img)
        anchor_kp, anchor_desc, anchor_name = kp, desc, frames[i].path.name
        consecutive_skips = 0

    if len(images) == 1:
        raise StitchError(
            "Only the first frame could be placed; no other frame registered to it. "
            "Check ordering/overlap, or try --detector sift and (for wide-angle "
            "footage) --undistort."
        )
    if skipped:
        report(total - 1, f"placed {len(images)}/{total} frames ({skipped} dropped)")

    return _compose_canvas(images, global_h, cfg)


def _compose_canvas(images, transforms, cfg: StitchConfig):
    """Warp all images by their global transforms onto one blended canvas."""
    import cv2
    import numpy as np

    # Find the bounding box of every warped frame to size the canvas.
    all_corners = []
    for img, h in zip(images, transforms):
        ih, iw = img.shape[:2]
        corners = np.float32([[0, 0], [iw, 0], [iw, ih], [0, ih]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, h))
    stacked = np.concatenate(all_corners, axis=0)
    x_min, y_min = stacked.min(axis=0).ravel()
    x_max, y_max = stacked.max(axis=0).ravel()

    offset = np.array(
        [[1, 0, -np.floor(x_min)], [0, 1, -np.floor(y_min)], [0, 0, 1]], dtype=np.float64
    )
    width = int(np.ceil(x_max - np.floor(x_min)))
    height = int(np.ceil(y_max - np.floor(y_min)))
    if width <= 0 or height <= 0:
        raise StitchError("Computed an empty canvas; registration likely diverged.")

    canvas = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)

    for img, h in zip(images, transforms):
        m = offset @ h
        warped = cv2.warpPerspective(img.astype(np.float32), m, (width, height))
        if cfg.blend == "feather":
            w = _feather_mask(img.shape[:2])
        else:  # overwrite: hard mask of valid pixels
            w = np.ones(img.shape[:2], dtype=np.float32)
        warped_w = cv2.warpPerspective(w, m, (width, height))[..., None]
        if cfg.blend == "overwrite":
            mask = warped_w > 0
            canvas = np.where(mask, warped, canvas)
            weight = np.where(mask, 1.0, weight)
        else:
            canvas += warped * warped_w
            weight += warped_w

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(weight > 0, canvas / np.maximum(weight, 1e-6), 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _feather_mask(shape):
    """Distance-to-edge weight so seams blend smoothly across overlaps."""
    import numpy as np

    h, w = shape
    ys = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    xs = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    mask = np.minimum.outer(ys, xs)
    peak = mask.max()
    if peak > 0:
        mask = mask / peak
    return mask + 1e-3  # keep strictly positive so edge pixels still contribute
