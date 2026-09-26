"""
Fit the camera->top-down field map from the ROBOT POINT CLOUD, with no clicking.

make_fieldmap.py asks you to place 9 landmarks by hand. This does the same job
automatically, using the fact that robots only ever drive on the carpet: pool every
logged robot ground point across every match and the outline of that cloud IS the
field boundary.

Measured on 112 matches / 2.43M points: 94.6% of robot positions land inside the
fitted field, against 63.3% for the old 720p calibration rescaled to 1080p. The old
one was not merely stale -- its top landmark row sat BELOW where robots actually
drive, which is impossible for a far sideline, so it was mis-placed to begin with.

Limits, stated plainly:
  - The boundary is where robots GO, not where the carpet ends. Robots rarely touch
    the extreme corners, so the fitted field is slightly inset. Paths are therefore
    correct in shape and relative position but very slightly scaled in.
  - Percentile edges mean ~1-2% of points sit outside by construction.
  - This never sees a real field landmark, so it cannot fix a systematic skew the
    point cloud shares. For metric-accurate work, place the 9 points by hand with
    make_fieldmap.py -- this is the good default, not the ground truth.

PyCharm: click Run.
"""

import csv
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
import topdown as td

RESULTS = os.path.join(HERE, "results")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
ONLY_AFTER = "17:00"  # only use trajectory.csv written after this time TODAY, so a
# stale 720p-geometry file can never poison the fit.
# "" = use every trajectory.csv regardless of age.
DEGREE = 2  # polynomial degree; 2 absorbs the barrel distortion
X_LO, X_HI = 0.3, 99.7  # percentiles defining the left/right field edges
Y_LO, Y_HI = 1.0, 99.0  # percentiles defining the far/near sidelines
# ===========================================================================


def load_cloud(only_after=ONLY_AFTER):
    cut = 0
    if only_after:
        cut = time.mktime(
            time.strptime(time.strftime("%Y-%m-%d") + " " + only_after, "%Y-%m-%d %H:%M")
        )
    xs, ys, used, skipped = [], [], 0, 0
    for p in sorted(glob.glob(os.path.join(RESULTS, "*", "trajectory.csv"))):
        if os.path.getmtime(p) < cut:
            skipped += 1
            continue
        used += 1
        with open(p) as f:
            for r in csv.DictReader(f):
                xs.append(float(r["cx"]))
                ys.append(float(r["cy"]))
    if skipped:
        print(f"  skipped {skipped} trajectory.csv older than {only_after} (stale geometry)")
    return np.asarray(xs), np.asarray(ys), used


def _y_edges(xs, ys, x, half=90):
    m = np.abs(xs - x) <= half
    if m.sum() < 300:
        return None, None
    return np.percentile(ys[m], Y_LO), np.percentile(ys[m], Y_HI)


def _x_edges(xs, ys, y, half=60):
    m = np.abs(ys - y) <= half
    if m.sum() < 300:
        return None, None
    return np.percentile(xs[m], X_LO), np.percentile(xs[m], X_HI)


def fit(xs, ys):
    """The 9 landmarks, in topdown.POLY_LANDMARKS order."""
    xL, xR = np.percentile(xs, X_LO), np.percentile(xs, X_HI)
    xm = (xL + xR) / 2
    yT, yB = _y_edges(xs, ys, xm)
    ym = (yT + yB) / 2
    xLl, xRr = _x_edges(xs, ys, ym)
    xL, xR = min(xL, xLl), max(xR, xRr)
    yTl, yBl = _y_edges(xs, ys, xL + 60)
    yTr, yBr = _y_edges(xs, ys, xR - 60)
    return [
        (xL, yTl),
        (xm, yT),
        (xR, yTr),
        (xL, (yTl + yBl) / 2),
        (xm, ym),
        (xR, (yTr + yBr) / 2),
        (xL, yBl),
        (xm, yB),
        (xR, yBr),
    ]


def main():
    xs, ys, used = load_cloud()
    if not len(xs):
        raise SystemExit("No trajectory points found -- run the pipeline first.")
    print(f"  {len(xs):,} robot ground points from {used} matches")
    pts = fit(xs, ys)
    for name, p in zip([n for n, _ in td.POLY_LANDMARKS], pts):
        print(f"   {name:3s} ({p[0]:7.1f},{p[1]:6.1f})")

    if os.path.exists(td.CALIB):
        bak = td.CALIB + ".bak"
        if not os.path.exists(bak):
            import shutil

            shutil.copy(td.CALIB, bak)
            print(f"  backed up previous calibration -> {bak}")
    td.save_fieldmap([(float(a), float(b)) for a, b in pts], DEGREE)

    fm = td.load_mapper()
    m = fm(np.stack([xs, ys], 1))
    inside = (m[:, 0] >= td.FX0) & (m[:, 0] <= td.FX1) & (m[:, 1] >= td.FY0) & (m[:, 1] <= td.FY1)
    print(f"\n  saved. {inside.mean() * 100:.1f}% of robot points land inside the field rectangle.")
    print(
        "  Sanity: this should be >90%. Much lower means the cloud is polluted by "
        "stale-geometry trajectories -- check ONLY_AFTER."
    )


if __name__ == "__main__":
    main()
