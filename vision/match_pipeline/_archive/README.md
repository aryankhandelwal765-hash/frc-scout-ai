# _archive

Superseded calibration scripts. Kept because the active code still refers to
them by name when explaining why it works the way it does, and because the
field maps they produced are what `calib.json` still contains for older events.

Nothing on the canonical path imports or invokes any of these. The canonical
calibration path is:

```
multi_video_apriltag_calib.py   # Phase 1 — AprilTags across ALL videos -> base field map
fit_2d_correction.py            # Phase 2 — 2D polynomial correction on top of it
```

both driven by `overnight_pipeline.py`. `field_calibrate.py` (still in
`match_pipeline/`) is the one-time interactive anchor picker and is NOT archived.

| Script | Why it is here |
|--------|----------------|
| `calib_solve.py` | Single-video AprilTag solve. `multi_video_apriltag_calib.py` does the same across every video, which is strictly better conditioned. |
| `calib_refine.py` | 1D `x_error(y)` polynomial refinement. `fit_2d_correction.py` fits a full 2D correction using the same trimming logic. `zones.py` still reads the 1D `x_correction` key from `calib.json` for backwards compatibility. |
| `autofit_fieldmap.py` | Inferred the field outline from the robot point cloud, no manual clicks. Superseded by the AprilTag solve, which is metric rather than inferred. |
| `calibrate_field.py` | Written to replace `autofit_fieldmap.py`; both were superseded by the AprilTag path before it became the default. |
| `make_fieldmap.py` | Manual 9-landmark placement. `autofit_fieldmap.py` automated it, then AprilTags replaced both. |
| `make_homography.py` | Manual 4-corner homography. Subsumed by the AprilTag ground homography. |
