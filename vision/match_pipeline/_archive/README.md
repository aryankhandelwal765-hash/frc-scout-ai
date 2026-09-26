# _archive

Calibration scripts that are not on the path currently producing the field map.
Kept because the active code still refers to them by name when explaining why it
works the way it does.

## What actually produces the field map today

`autofit_fieldmap.py` (in `match_pipeline/`, not here). It fits a degree-2
polynomial through 9 landmarks inferred from the robot point cloud and writes it
to `calib.json["fieldmap"]`. That is the map `topdown.load_mapper()` returns and
every downstream script uses.

The AprilTag route was **attempted and rejected**, not adopted:

```
calib.json["ground_homography"]:  n_tags=16, n_corners=64,
                                  reproj_px=239.5, "bad": true
calib.json["correction_2d"]:      absent — fit_2d_correction.py never completed
```

A 239-pixel reprojection error is not usable, so `multi_video_apriltag_calib.py`
never replaced the polynomial (it would have left a `fieldmap_poly_backup` key
behind; there is none). Phases 2-4 of `overnight_pipeline.py` have never
completed. Both scripts are still in `match_pipeline/` because the intent is to
get the AprilTag path converging — see Roadmap in the root README.

## Why each of these is here

| Script | Why it is here |
|--------|----------------|
| `calib_solve.py` | Single-video AprilTag solve. Superseded in intent by `multi_video_apriltag_calib.py`, which uses every video — though neither has converged yet. |
| `calib_refine.py` | 1D `x_error(y)` polynomial refinement. `fit_2d_correction.py` is the intended successor. `zones.py` still reads the 1D `x_correction` key from `calib.json` if present, for backwards compatibility. |
| `calibrate_field.py` | An attempt to replace `autofit_fieldmap.py` with a hand-clicked fit. It writes `fieldmap_autofit_backup` when it runs; that key is absent from `calib.json`, so it did not produce the live map. |
| `make_fieldmap.py` | Manual 9-landmark placement. `autofit_fieldmap.py` automates the same fit and is what ran. |
| `make_homography.py` | Manual 4-corner homography. The hand-clicked result is still parked in `calib.json["fieldmap_handclicked_backup"]`, but `load_mapper()` prefers `fieldmap`. |

`field_calibrate.py` is **not** archived — it is the one-time interactive anchor
picker that `auto_paths.py` and `preferred_paths.py` tell the user to run.
`make_zones.py` is not archived either; it authors zone config into `calib.json`.
