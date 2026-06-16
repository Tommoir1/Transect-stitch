# Transect Stitch

Stitch imagery collected along a survey **transect** into a single long mosaic.

Built for cameras mounted on an **Uncrewed Surface Vehicle (USV)** (or any moving
platform) that captures either a **time-lapse** or a folder of **individual images**
while travelling in a roughly straight line. The vehicle may rock, yaw, and surge in
waves, so frames are registered to each other with robust feature matching rather than
assuming a fixed offset. When images carry **GPS / timestamp EXIF metadata**, that
metadata is used to order frames correctly and to sanity-check the geometry.

---

## What it does

1. **Discovers** images from a folder (or an explicit list / glob).
2. **Orders** them by GPS timestamp (falling back to EXIF capture time, then filename).
3. **Registers** consecutive frames with ORB/SIFT feature matching + RANSAC, estimating
   a partial-affine transform (translation + rotation + scale) so platform rock/yaw is
   absorbed instead of smearing the mosaic.
4. **Accumulates** transforms and **warps** every frame onto one growing canvas.
5. **Blends** overlaps (feathering by default) and writes a single long image.

A `--dry-run` mode reports the discovered/ordered frames and their GPS track without
doing any pixel work, which is handy for checking a dataset before a long stitch.

## Install

```bash
git clone https://github.com/Tommoir1/transect-stitch.git
cd transect-stitch
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs two commands: `transect-stitch` (CLI) and `transect-stitch-gui`
(desktop GUI). The library is also importable as `transect_stitch`.

> The GUI uses **Tkinter**, which ships with most Python installs. On Linux you may
> need the system package first: `sudo apt install python3-tk`.

## GUI

```bash
transect-stitch-gui
```

A simple window where you can:

- **Add Images…** or **Add Folder…**, then see them in transect order (the list
  updates live; `[gps]`/`[exif]` tags show where each frame's order came from).
- Stitch a **single mosaic** from all the images, **or**
- Run in **batch** mode — "one mosaic every N images" (e.g. every 40) — writing one
  output file per group into a folder you pick.
- Tune the detector, blending, downscale (`Max dim`), and **Use every Nth image**
  (handy to thin dense time-lapses).

Stitching runs on a background thread with a progress bar and log, and in batch mode a
group that can't be registered is skipped (logged) rather than aborting the whole run.

## Usage (CLI)

```bash
# Stitch every image in a folder, ordered by GPS time, into one mosaic
transect-stitch ./survey_images -o mosaic.jpg

# Use a glob, choose the feature detector and overlap blending
transect-stitch "frames/*.png" -o mosaic.png --detector sift --blend feather

# Inspect a dataset without stitching (ordering + GPS track + gaps)
transect-stitch ./survey_images --dry-run

# Order strictly by filename (e.g. frame_0001.jpg) when metadata is missing
transect-stitch ./frames -o out.jpg --order filename

# Thin a dense time-lapse: use every 3rd frame
transect-stitch ./frames -o out.jpg --stride 3

# Batch mode: one mosaic per 40 frames, written into ./mosaics/
transect-stitch ./frames -o ./mosaics --batch-size 40

# Hard case: GoPro / wide-angle underwater footage (fisheye + low contrast)
transect-stitch ./frames -o ./mosaics --batch-size 40 \
    --undistort -0.3 --detector sift --max-features 8000
```

## Hard imagery (wide-angle / underwater)

Action-cam footage (GoPro etc.) and underwater scenes are the difficult case,
and three options exist specifically for them:

- `--undistort K1` — wide-angle/fisheye lenses curve straight lines, which the
  affine registration model can't fit, so RANSAC throws out almost every match.
  A negative `K1` (start around `-0.3` for a GoPro) straightens the frame so
  pairs actually register. `0` (default) leaves frames untouched.
- **CLAHE** (on by default; disable with `--no-clahe`) lifts local contrast on
  hazy/low-contrast underwater frames, surfacing far more features.
- `--max-skip N` (default 5) — a single blurry or textureless frame is now
  *dropped* and the mosaic continues from the last good frame, instead of the
  whole run aborting. Failures report how many matches were geometrically
  consistent, so you can tell "no overlap" from "too much lens distortion".

If a mosaic still won't form, try a lower `--stride`, `--detector sift`, or a
slightly stronger `--undistort`.

Run `transect-stitch --help` for the full option list.

## How ordering works

For each image the loader tries, in order:

1. **GPS timestamp** — `GPSDateStamp` + `GPSTimeStamp` EXIF tags (UTC, most reliable
   across devices).
2. **Capture time** — `DateTimeOriginal` EXIF tag.
3. **Filename** — natural sort, so `frame_2.jpg` sorts before `frame_10.jpg`.

GPS coordinates (when present) are also extracted so `--dry-run` can print the track and
flag large jumps between consecutive frames (possible dropouts or out-of-order data).

## Limitations / notes

- Designed for **linear transects**, not 360° panoramas. Scenes need enough texture and
  overlap (~30%+) between consecutive frames for feature matching to lock on.
- Very low-texture water with no features (open ocean, no seabed visible) is the hard
  case — there simply isn't anything to match. GPS-only placement is on the roadmap.
- Heavy parallax (close foreground + far background) can cause seams; reduce by keeping
  the camera distance roughly constant.

## License

MIT — see [LICENSE](LICENSE).
