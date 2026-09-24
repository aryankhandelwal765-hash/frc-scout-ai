"""
Refine the field map using the ROBOT STARTING LINE as ground truth.

THE PROBLEM THIS FIXES. The active field map gets the middle of the field right and
the edges wrong. Measured against the starting line, the X error swings from about
-60in at one guardrail to +56in at the other -- a 116in spread across the width.
Zone bands (BUMP, TRENCH, HUB) are only 47in deep, so that skew moves a robot by
more than a whole zone. It is why the coarse questions work (which half of the field
a robot is on: 3.0% rule violations) while the fine ones do not (bump/trench
split-half reliability -0.38 to +0.30, against 0.72-0.88 for a human scout).

THE GROUND TRUTH. The manual requires a ROBOT's BUMPERS to overlap its ROBOT
STARTING LINE at MATCH start, and auto_start.py knows which frame that is. So at
that frame every tracked robot sits at a KNOWN X -- 158.6in from its own wall. That
yields ~300 observations spanning the full width, which is far more constraint than
12 hand clicks or the 6 AprilTags visible in the top view, and it needs nothing from
the user.

WHAT IT CAN AND CANNOT DO. Ground truth exists at only two X values (the two
starting lines), so this fits a correction to X as a function of Y and applies it
everywhere. The dominant measured error is exactly that shape, so it addresses the
bulk of it -- but a residual error that varies along X cannot be seen by this method
and is not corrected.

VALIDATION IS OBJECTIVE, NOT AESTHETIC. Success is not "the paths look better". It
is that bump/trench split-half reliability rises toward the human's 0.72-0.88, and
that the CENTER LINE rule-violation rate does not get worse. Both are printed.

PyCharm: click Run.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
import auto_paths as ap
import topdown as td
import topdown_paths as tp
import zones

RESULTS = os.path.join(HERE, "results")
CALIB = os.path.join(HERE, "calib.json")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
DEGREE = 2          # polynomial in Y for the X correction
TRIM = 0.20         # trim this fraction from each tail per Y-bin before fitting,
                    # so a robot that already drove off the line cannot drag the fit
MIN_TRACK = 60      # ignore short flicker tracks
FRAME_TOL = 4       # samples this close to the detected start frame count
APPLY = True        # False = report only, write nothing
# ===========================================================================


def observations(mapper):
    """(y_in, x_err_in) for every robot sitting on a starting line at match start."""
    obs = []
    for apf in sorted(glob.glob(os.path.join(RESULTS, "*", "auto_paths.json"))):
        if not tp._fresh(apf):
            continue
        match = os.path.basename(os.path.dirname(apf))
        sf = ap._detected_start(match)
        if sf is None:
            continue
        sf = max(0, sf)
        tracks = ap._load_trajectory(match)
        if not tracks:
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
            fi = zones.to_field_inches(mapper([(r[3], r[4])]))[0]
            al = ap._voted_alliance(match, tid, tr)
            own_left = (al == "blue") == blue_left
            expect = zones.BAND_NEAR if own_left else zones.FIELD_LEN - zones.BAND_NEAR
            obs.append((fi[1], fi[0] - expect))
    return np.array(obs)


def fit(obs, degree=DEGREE, trim=TRIM, bins=8):
    """Robust polynomial x_err(y). Binned + trimmed so outliers cannot steer it."""
    y, e = obs[:, 0], obs[:, 1]
    xs, ys, ws = [], [], []
    edges = np.linspace(0, zones.FIELD_WID, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b)
        if m.sum() < 8:
            continue
        v = np.sort(e[m])
        k = int(len(v) * trim)
        v = v[k:len(v) - k] if len(v) - 2 * k >= 3 else v
        xs.append((a + b) / 2.0)
        ys.append(float(np.median(v)))
        ws.append(m.sum())
    if len(xs) < degree + 1:
        raise SystemExit("not enough populated Y bins to fit a correction")
    return np.polyfit(xs, ys, degree, w=np.sqrt(ws)), list(zip(xs, ys, ws))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", default=APPLY)
    a = p.parse_args()

    mapper = td.load_mapper()
    if mapper is None:
        raise SystemExit("No field map -- run autofit_fieldmap.py first.")

    obs = observations(mapper)
    print(f"{len(obs)} robots observed on a ROBOT STARTING LINE at match start")
    coef, binned = fit(obs)
    print("\n  Y bin centre   median X-error   n")
    for cy, ce, n in binned:
        print(f"     {cy:6.0f}       {ce:+8.0f} in     {n}")
    span = max(e for _c, e, _n in binned) - min(e for _c, e, _n in binned)
    print(f"\n  correction spans {span:.0f} in across the width "
          f"(zone bands are 47 in deep)")

    resid = obs[:, 1] - np.polyval(coef, obs[:, 0])
    print(f"  median |error| before {np.median(np.abs(obs[:, 1])):5.1f} in"
          f"   after {np.median(np.abs(resid)):5.1f} in")

    if not a.apply:
        print("\n[report only] nothing written")
        return

    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    calib["x_correction"] = {
        "coef": [float(c) for c in coef], "degree": DEGREE,
        "n_obs": int(len(obs)),
        "note": ("x_field_in -= polyval(coef, y_field_in); fitted from robots on "
                 "the ROBOT STARTING LINE at match start (manual: BUMPERS must "
                 "overlap it)"),
    }
    json.dump(calib, open(CALIB, "w"), indent=2)
    print(f"\nsaved x_correction into {CALIB}")
    print("zones.to_field_inches() applies it automatically from now on.")
    print("\nNow re-run, in order:")
    print("  teleop_paths.py   scout_table.py   validate_paths.py")
    print("then re-check bump/trench split-half reliability -- that is the test.")


if __name__ == "__main__":
    main()
