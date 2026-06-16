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
    min_matches: int = 12  # min good matches to trust a pair
    ransac_thresh: float = 4.0  # reprojection tolerance (px)
    blend: str = "feather"  # "feather" | "overwrite"
    max_dim: int = 0  # if >0, downscale each frame so max(h,w) <= max_dim


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


def _load_image(path: Path, cfg: StitchConfig):
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise StitchError(f"Could not read image: {path}")
    if cfg.max_dim > 0:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > cfg.max_dim:
            scale = cfg.max_dim / longest
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


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
    """Partial-affine 2x3 mapping points in frame2 -> frame1, or None."""
    import cv2
    import numpy as np

    if len(matches) < cfg.min_matches:
        return None
    src = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh
    )
    if matrix is None or inliers is None or int(inliers.sum()) < cfg.min_matches:
        return None
    return matrix


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
    prev_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_desc = detector.detectAndCompute(prev_gray, None)

    global_h = [np.eye(3, dtype=np.float64)]  # frame i -> global
    images = [base]

    if total == 1:
        return base

    cur_global = np.eye(3, dtype=np.float64)
    for i in range(1, total):
        report(i, f"registering {frames[i].path.name}")
        img = _load_image(frames[i].path, cfg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detector.detectAndCompute(gray, None)

        matches = _good_matches(matcher, prev_desc, desc, cfg.ratio)
        affine = _estimate_pairwise(prev_kp, kp, matches, cfg)
        if affine is None:
            raise StitchError(
                f"Could not register frame {i} ({frames[i].path.name}) to the "
                f"previous frame: only {len(matches)} good matches "
                f"(need >= {cfg.min_matches}). The overlap or texture is too low."
            )

        # affine maps current -> previous; compose onto previous global transform.
        cur_global = cur_global @ _to_3x3(affine)
        global_h.append(cur_global.copy())
        images.append(img)

        prev_kp, prev_desc, prev_gray = kp, desc, gray

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
