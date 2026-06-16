# Examples

Drop a folder of transect frames here (or point the CLI anywhere) and try:

```bash
# Inspect ordering + GPS track without stitching
transect-stitch ./examples/my_survey --dry-run

# Stitch at half-ish resolution for a quick preview
transect-stitch ./examples/my_survey -o preview.jpg --max-dim 1600

# Full-resolution mosaic with SIFT (more robust on low-texture scenes)
transect-stitch ./examples/my_survey -o mosaic.jpg --detector sift
```

## Tips for good USV transects

- **Overlap**: aim for ~30–60% overlap between consecutive frames so there are
  shared features to match. Faster vehicle or slower frame rate => less overlap.
- **Texture**: feature matching needs detail. Seabed, kelp, structure, and shoreline
  stitch well; flat open water with no features is the hard case.
- **Exposure**: lock exposure/white balance if you can — big brightness swings between
  frames make seams more visible (feather blending helps, but can't fix everything).
- **GPS**: keep GPS/EXIF timestamps enabled so `--order auto` can sequence frames even
  if filenames are not monotonic.
