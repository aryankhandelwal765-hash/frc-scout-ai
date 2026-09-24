"""
Phase 2 of the overnight pipeline: fit a 2D polynomial correction on top of
the Phase 1 camera pose.

Phase 1 (multi_video_apriltag_calib.py) dramatically improves the base fieldmap.
Phase 2 collects the remaining systematic error from robot starting-line observations
and fits a degree-2 polynomial in BOTH (raw_X, raw_Y) -- not just in Y as the old
calib_refine.py did.

Why 2D instead of 1D: the Phase 1 ground_homography corrects the large camera-pose
error. What remains is a small residual that varies across BOTH axes, because:
  - The broadcaster may not place the camera EXACTLY where Phase 1 assumes
  - Barrel distortion is only approximated by k1/k2 from camera_calib.json
  - The floor is not perfectly flat (bumps, field tiles)
A degree-2 polynomial in (X, Y) fits this residual with 6 parameters -- plenty
of freedom, no risk of overfitting 300+ starting-line observations.

Also fits a Y correction using field-boundary soft constraints: any robot observed
with mapped Y < 0 or Y > FIELD_WID is clamped and contributes a boundary pair.

Writes calib.json["correction_2d"]:
  {coef_x: [...], coef_y: [...], degree: 2, n_obs: N}

zones.to_field_inches() applies this automatically when it exists.

Run: python fit_2d_correction.py   (after multi_video_apriltag_calib.py)
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import auto_paths as ap
import topdown as td
import topdown_paths as tp
import zones

RESULTS = os.path.join(HERE, "results")
CALIB = os.path.join(HERE, "calib.json")

DEGREE = 2          # polynomial degree; 6 coefficients for 2D degree-2
TRIM = 0.15         # trim this fraction from each Y-bin's tail before fitting
MIN_TRACK = 60      # ignore short tracks (robot left frame early / ID flicker)
FRAME_TOL = 4       # frames around detected start frame that count as "at start"
BOUNDARY_WEIGHT = 0.3  # weight of field-boundary soft constraints vs starting-line


def _poly2d_features(X, Y, degree=2):
    """Build polynomial feature matrix. Degree 2 -> 6 columns: [1,x,y,x2,xy,y2]."""
    x, y = np.asarray(X, float), np.asarray(Y, float)
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append(x ** (d - i) * y ** i)
    return np.column_stack(cols)


def _ridge(A, b, w=None, reg=1.0):
    """Weighted ridge regression: min ||W(Ax - b)||^2 + reg*||x||^2."""
    if w is not None:
        W = np.sqrt(np.asarray(w, float))
        A = A * W[:, None]
        b = b * W
    ATA = A.T @ A
    ATA[np.diag_indices_from(ATA)] += reg
    return np.linalg.solve(ATA, A.T @ b)


def starting_line_obs(mapper):
    """(raw_X, raw_Y, true_X) for robots at their starting line at match start.

    Uses the same logic as calib_refine.py but with the Phase 1 mapper so the
    residual being fitted is small and well-conditioned.
    """
    obs = []
    for apf in sorted(glob.glob(os.path.join(RESULTS, "*", "auto_paths.json"))):
        if not tp._fresh(apf):
            continue
        match = os.path.basename(os.path.dirname(apf))
        sf = ap._detected_start(match)
        if sf is None:
            continue
        sf = max(0, sf)
        try:
            tracks = ap._load_trajectory(match)
        except SystemExit:
            continue
        try:
            blue_left = tp._match_blue_left(json.load(open(apf))["teams"])
        except Exception:
            continue
        for tid, tr in tracks.items():
            if len(tr) < MIN_TRACK:
                continue
            near = [r for r in tr if abs(r[0] - sf) <= FRAME_TOL]
            if not near:
                continue
            r = min(near, key=lambda z: abs(z[0] - sf))
            canvas_xy = mapper([(r[3], r[4])])
            fi = zones.to_field_inches(canvas_xy)[0]
            raw_x, raw_y = fi[0], fi[1]
            al = ap._voted_alliance(match, tid, tr)
            own_left = (al == "blue") == blue_left
            true_x = zones.BAND_NEAR if own_left else zones.FIELD_LEN - zones.BAND_NEAR
            obs.append((raw_x, raw_y, true_x))
    return np.array(obs)


def boundary_obs(mapper, clamp_margin=15.0):
    """Soft boundary pairs: robots past the guardrails are clamped to the field edge.

    A robot at raw_Y < -clamp_margin is clearly past the guardrail -- its true Y
    is (approximately) 0. Same at the far edge.  These pairs constrain the Y fit
    in regions where no starting-line robots sit.
    """
    pairs = []
    for tcsv in sorted(glob.glob(os.path.join(RESULTS, "*", "trajectory.csv"))):
        import csv
        with open(tcsv) as f:
            for row in csv.DictReader(f):
                cx, cy = float(row["cx"]), float(row["cy"])
                canvas_xy = mapper([(cx, cy)])
                fi = zones.to_field_inches(canvas_xy)[0]
                raw_x, raw_y = fi[0], fi[1]
                if raw_y < -clamp_margin:
                    pairs.append((raw_x, raw_y, raw_x, 0.0))        # true Y = 0
                elif raw_y > zones.FIELD_WID + clamp_margin:
                    pairs.append((raw_x, raw_y, raw_x, zones.FIELD_WID))  # true Y = FIELD_WID
    return pairs


def fit(obs_3col, boundary_pairs, degree=DEGREE, trim=TRIM, bins=8, reg=0.5):
    """Fit coef_x and coef_y polynomials.

    obs_3col: (N, 3) array of (raw_X, raw_Y, true_X).  Only X supervised here.
    boundary_pairs: list of (raw_X, raw_Y, true_X, true_Y) from field edges.
    """
    raw_x, raw_y, true_x = obs_3col[:, 0], obs_3col[:, 1], obs_3col[:, 2]
    err_x = true_x - raw_x

    # Robust binned median for X correction (same trimming as calib_refine.py)
    xs_b, ex_b, w_b = [], [], []
    edges = np.linspace(0, zones.FIELD_WID, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (raw_y >= a) & (raw_y < b)
        if m.sum() < 8:
            continue
        v = np.sort(err_x[m])
        k = int(len(v) * trim)
        v = v[k:len(v) - k] if len(v) - 2 * k >= 3 else v
        xs_b.append((a + b) / 2.0)
        ex_b.append(float(np.median(v)))
        w_b.append(float(m.sum()))

    if len(xs_b) < degree + 1:
        raise SystemExit("Not enough Y-bin observations to fit X correction. "
                         "Run batch_auto.py first.")

    # Fit coef_x: poly2d(raw_X, raw_Y) -> X error
    # Primary signal: binned medians along Y (X held at starting-line X ≈ 158.6 or 492.6)
    # We evaluate the polynomial at raw_X=325.6 (midfield) for the binned Y centres
    # so the fit is anchored in the measured Y range.
    mid_x = zones.FIELD_LEN / 2.0
    A_b = _poly2d_features([mid_x] * len(xs_b), xs_b, degree)
    coef_x = _ridge(A_b, np.array(ex_b), w=np.array(w_b), reg=reg)

    # Refine coef_x using all individual observations (unweighted beyond the binning)
    A_all = _poly2d_features(raw_x, raw_y, degree)
    pred_all = A_all @ coef_x
    resid_all = err_x - pred_all
    # downweight outliers (>3σ)
    sigma = np.std(resid_all)
    weights = np.where(np.abs(resid_all) < 3 * sigma, 1.0, 0.1)
    coef_x = _ridge(A_all, err_x, w=weights, reg=reg)

    # Fit coef_y: poly2d(raw_X, raw_Y) -> Y error, from boundary constraints only
    if boundary_pairs:
        bp = np.array(boundary_pairs)
        bx, by, _, true_y = bp[:, 0], bp[:, 1], bp[:, 2], bp[:, 3]
        err_y = true_y - by
        A_y = _poly2d_features(bx, by, degree)
        w_y = np.full(len(bx), BOUNDARY_WEIGHT)
        coef_y = _ridge(A_y, err_y, w=w_y, reg=reg)
    else:
        # No boundary data -- Y correction is zero (Phase 1 handles Y)
        coef_y = np.zeros(A_b.shape[1])

    return coef_x.tolist(), coef_y.tolist()


def report(mapper, coef_x, coef_y, obs_3col, degree=DEGREE):
    raw_x, raw_y, true_x = obs_3col[:, 0], obs_3col[:, 1], obs_3col[:, 2]
    A = _poly2d_features(raw_x, raw_y, degree)
    dx = A @ np.array(coef_x)
    err_before = np.abs(true_x - raw_x)
    err_after = np.abs(true_x - (raw_x + dx))
    print(f"\n  Observations: {len(raw_x)} robots on a starting line")
    print(f"  X error before: median={np.median(err_before):.1f}in  "
          f"p90={np.percentile(err_before, 90):.1f}in")
    print(f"  X error after:  median={np.median(err_after):.1f}in  "
          f"p90={np.percentile(err_after, 90):.1f}in")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-apply", action="store_true")
    a = p.parse_args()

    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    if "fieldmap" not in calib:
        raise SystemExit("No fieldmap in calib.json -- run multi_video_apriltag_calib.py first")

    # Temporarily clear any existing correction so observations come from the raw Phase 1 mapper
    _saved_corr = calib.pop("correction_2d", None)
    _saved_xcorr = calib.pop("x_correction", None)
    json.dump(calib, open(CALIB, "w"), indent=2)
    zones._XCORR = None  # bust cache

    try:
        mapper = td.load_mapper()
        if mapper is None:
            raise SystemExit("load_mapper() returned None")

        print("Collecting starting-line observations ...")
        obs = starting_line_obs(mapper)
        if len(obs) < 20:
            raise SystemExit(f"Only {len(obs)} starting-line observations -- "
                             "run batch_auto.py on at least a few matches first")

        print("Collecting field-boundary constraints ...")
        bp = boundary_obs(mapper)
        print(f"  {len(bp)} boundary pairs (robots past guardrails)")

        print("Fitting 2D polynomial correction ...")
        coef_x, coef_y = fit(obs, bp, degree=DEGREE)
        report(mapper, coef_x, coef_y, obs)

    finally:
        # Restore or write correction
        calib = json.load(open(CALIB))
        if _saved_xcorr:
            calib["x_correction"] = _saved_xcorr
        zones._XCORR = None

    if a.no_apply:
        print("\n[--no-apply] nothing written")
        return

    calib["correction_2d"] = {
        "coef_x": coef_x,
        "coef_y": coef_y,
        "degree": DEGREE,
        "n_obs": int(len(obs)),
        "note": "poly2d(raw_X, raw_Y) -> (dX, dY) correction; applied by zones.to_field_inches()",
    }
    json.dump(calib, open(CALIB, "w"), indent=2)
    print(f"\nWritten calib['correction_2d'] to {CALIB}")
    print("zones.to_field_inches() will apply it automatically from now on.")
    print("\nNext: smooth_trajectories.py")


if __name__ == "__main__":
    main()
