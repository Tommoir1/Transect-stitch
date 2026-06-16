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

This installs the `transect-stitch` console command (and `python -m transect_stitch`).

## Usage

```bash
# Stitch every image in a folder, ordered by GPS time, into one mosaic
transect-stitch ./survey_images -o mosaic.jpg

# Use a glob, choose the feature detector and overlap blending
transect-stitch "frames/*.png" -o mosaic.png --detector sift --blend feather

# Inspect a dataset without stitching (ordering + GPS track + gaps)
transect-stitch ./survey_images --dry-run

# Order strictly by filename (e.g. frame_0001.jpg) when metadata is missing
transect-stitch ./frames -o out.jpg --order filename
```

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
