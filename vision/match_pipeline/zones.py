"""
Named FIELD ZONES in real field coordinates, and point/path -> zone lookup.

This is what unblocks the scouting columns that vision could not fill: bump and
trench usage, and the path_sequence a human scout writes by hand.

WHY THIS CAN NOW BE BUILT. The manual gives every structure's SIZE but places them
only in figures -- section 5 never states how far a BUMP or TRENCH sits along the
field. The official AprilTag layout closes that gap, because tags are mounted ON
those structures at published coordinates:

    HUB tags        X 160-210 (blue) / 450-490 (red)   -> confirms the HUB occupies
                    158.6..205.6in from its wall, i.e. its near face is exactly on
                    the ROBOT STARTING LINE
    TRENCH tags     X ~180 / ~470, Y ~30 and ~290      -> trenches hug the guardrails
                    inside the same band as the HUB
    TOWER/OUTPOST   X 0 and 650                        -> the ALLIANCE WALLS

So the layout below is measured, not guessed, apart from the caveats at the bottom.

COORDINATES (inches): X along the field length, 0 = BLUE ALLIANCE WALL,
651.2 = RED ALLIANCE WALL. Y across the width, 0..317.7. Zone names follow the
scouting sheet's vocabulary, which matches the manual's own terms.

KNOWN INCONSISTENCY, stated rather than smoothed over: the manual's zone depths do
not tile the field (158.6 + 283 + 158.6 = 600.2 vs a stated 651.2 length, 51in
unaccounted). The structure band is anchored on the AprilTag positions, which are
unambiguous, so that discrepancy lands in the open neutral area rather than in the
structures teams actually care about.
"""

import os

FIELD_LEN = 651.2
FIELD_WID = 317.7

# structure band, confirmed by AprilTag X positions
BAND_NEAR = 158.6                 # ROBOT STARTING LINE / HUB near face
BAND_FAR = 205.6                  # HUB far face (47in deep)
RED_BAND_FAR = FIELD_LEN - BAND_NEAR      # 492.6
RED_BAND_NEAR = FIELD_LEN - BAND_FAR      # 445.6

# lateral bands across the width at the structure band (manual 5.4/5.5/5.6)
HUB_Y0, HUB_Y1 = FIELD_WID / 2 - 23.5, FIELD_WID / 2 + 23.5    # 135.35 .. 182.35
BUMP_Y0, BUMP_Y1 = HUB_Y0 - 73.0, HUB_Y1 + 73.0                # 62.35 .. 255.35

# "TOP"/"BOTTOM" are as seen in the broadcast: TOP = far guardrail (Y=0).
NEUTRAL_ROWS = [("Top", 0.0, FIELD_WID / 3),
                ("Mid", FIELD_WID / 3, 2 * FIELD_WID / 3),
                ("Bot", 2 * FIELD_WID / 3, FIELD_WID)]


def _rect(x0, y0, x1, y1):
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def build_zones():
    """name -> (x0, y0, x1, y1) in field inches. Checked in priority order, so
    structures win over the broad zones they sit inside."""
    z = {}
    for tag, near, far in (("Blue", BAND_NEAR, BAND_FAR),
                           ("Red", RED_BAND_FAR, RED_BAND_NEAR)):
        z[f"{tag} Hub"] = _rect(near, HUB_Y0, far, HUB_Y1)
        z[f"Top {tag} Bump"] = _rect(near, BUMP_Y0, far, HUB_Y0)
        z[f"Bot {tag} Bump"] = _rect(near, HUB_Y1, far, BUMP_Y1)
        z[f"Top {tag} Trench"] = _rect(near, 0.0, far, BUMP_Y0)
        z[f"Bot {tag} Trench"] = _rect(near, BUMP_Y1, far, FIELD_WID)

    # alliance zones: the band between a wall and its ROBOT STARTING LINE.
    # DEPOT and OUTPOST sit against the wall at opposite corners (manual 5.7/5.9.2,
    # corner confirmed by OUTPOST tag Y of ~30 and ~290).
    for tag, wx, ix in (("Blue", 0.0, BAND_NEAR), ("Red", FIELD_LEN, RED_BAND_FAR)):
        z[f"{tag} Depot"] = _rect(wx, 0.0, ix, FIELD_WID / 3)
        z[f"{tag} Center"] = _rect(wx, FIELD_WID / 3, ix, 2 * FIELD_WID / 3)
        z[f"{tag} Outpost"] = _rect(wx, 2 * FIELD_WID / 3, ix, FIELD_WID)

    # open neutral area, 3 rows x 2 columns either side of the CENTER LINE
    mid = FIELD_LEN / 2
    for row, y0, y1 in NEUTRAL_ROWS:
        z[f"Neutral {row}-Left"] = _rect(BAND_FAR, y0, mid, y1)
        z[f"Neutral {row}-Right"] = _rect(mid, y0, RED_BAND_NEAR, y1)
    return z


ZONES = build_zones()
# structures are checked first so a point inside the HUB is not reported as the
# broad zone that geometrically contains it
PRIORITY = ([k for k in ZONES if "Hub" in k or "Bump" in k or "Trench" in k]
            + [k for k in ZONES if k not in
               [k2 for k2 in ZONES if "Hub" in k2 or "Bump" in k2 or "Trench" in k2]])


def zone_at(x_in, y_in):
    """Zone name for a field position in inches, or None if off-field."""
    for name in PRIORITY:
        x0, y0, x1, y1 = ZONES[name]
        if x0 <= x_in <= x1 and y0 <= y_in <= y1:
            return name
    return None


def zone_sequence(points_in, min_dwell=3):
    """[(x,y), ...] in field inches -> the ordered list of zones visited.

    Consecutive repeats collapse, and a zone must be held for `min_dwell` samples
    to be recorded -- otherwise clipping a corner registers as a visit and the
    sequence fills with noise a human scout would never write down.
    """
    seq, run, last = [], 0, None
    for x, y in points_in:
        zn = zone_at(x, y)
        if zn == last:
            run += 1
        else:
            last, run = zn, 1
        if zn and run == min_dwell and (not seq or seq[-1] != zn):
            seq.append(zn)
    return seq


def used(points_in, kind):
    """Did this path enter any zone of a kind ('Bump', 'Trench', 'Hub')?"""
    return any(kind in (zone_at(x, y) or "") for x, y in points_in)


_XCORR = None
_CORR2D = None


def _x_correction():
    """Polynomial x-error(y) fitted by calib_refine.py, or None."""
    global _XCORR
    if _XCORR is None:
        import json
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")
        try:
            _XCORR = (json.load(open(p)).get("x_correction") or {}).get("coef") or []
        except Exception:
            _XCORR = []
    return _XCORR


def _2d_correction():
    """2D polynomial correction fitted by fit_2d_correction.py, or None."""
    global _CORR2D
    if _CORR2D is None:
        import json
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")
        try:
            _CORR2D = json.load(open(p)).get("correction_2d") or {}
        except Exception:
            _CORR2D = {}
    return _CORR2D or None


def _apply_2d(x, y, corr):
    """Apply correction_2d to (x, y) arrays, return (corr_x, corr_y)."""
    import numpy as np
    deg = corr["degree"]
    cols = [np.ones_like(x)]
    for d in range(1, deg + 1):
        for i in range(d + 1):
            cols.append(x ** (d - i) * y ** i)
    A = np.column_stack(cols)
    return x + A @ np.array(corr["coef_x"]), y + A @ np.array(corr["coef_y"])


def to_field_inches(xy_canvas):
    """topdown canvas px -> field inches. Keeps callers from re-deriving the scale.

    Applies correction in priority order:
      1. correction_2d (Phase 2, 2D polynomial) if present
      2. x_correction (calib_refine.py 1D poly) if present
      3. No correction (raw scale only)
    """
    import topdown as td
    import numpy as np
    a = np.asarray(xy_canvas, float)
    sx = FIELD_LEN / (td.FX1 - td.FX0)
    sy = FIELD_WID / (td.FY1 - td.FY0)
    x = (a[:, 0] - td.FX0) * sx
    y = (a[:, 1] - td.FY0) * sy
    corr2d = _2d_correction()
    if corr2d:
        x, y = _apply_2d(x, y, corr2d)
    else:
        c = _x_correction()
        if c:
            x = x - np.polyval(c, y)
    return np.stack([x, y], 1)


if __name__ == "__main__":
    print(f"{len(ZONES)} zones on a {FIELD_LEN} x {FIELD_WID} in field\n")
    for n in PRIORITY:
        x0, y0, x1, y1 = ZONES[n]
        print(f"  {n:22s} X {x0:6.1f}..{x1:6.1f}   Y {y0:6.1f}..{y1:6.1f}")
