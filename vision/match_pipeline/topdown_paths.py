"""
Draw each robot's AUTO path on the top-down field -- the lined bird's-eye book.

Per match: takes the classified auto paths (results/<match>/auto_paths.json), pushes
each robot's ground track through the camera->field homography, and draws one clean
line per bot on a top-down field, coloured by alliance and labelled by team number.

Output -> results/<match>/topdown_auto.png
       -> topdown_paths/<team>.png        (each team's auto across all its matches)
       -> topdown_paths/_all.png          (one match's six, or an index)

Needs: a field map (autofit_fieldmap.py) + auto_paths.json (batch_auto.py).

  python topdown_paths.py --match 2026hop_qm27      # one match
  python topdown_paths.py                            # all matches + per-team aggregate
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

import topdown as td
import zones

try:
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    _SCIPY = True
except ImportError:
    _SCIPY = False

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(HERE, "topdown_paths")

# ===========================================================================
# EDIT ME (PyCharm one-click). CLI args override.
# ===========================================================================
MATCH = "2026hop_qm32"   # one match; "" = every match + per-team aggregates
MIN_MATCHES = 2           # a team needs this many autos to get an aggregate page
STALE_MAX_X = 1300        # a file whose widest path point is below this is in the old
                          # 1280-wide crop, so the 1080p field map turns it to
                          # nonsense. The per-team aggregate POOLS matches, so one
                          # stale file corrupts that whole team's page. 0 = no filter.
# ===========================================================================

# how the team number was determined -- same colour language as render_autopaths.py
SRC_COL = {"read": (60, 200, 60), "elim": (190, 160, 40),
           "rank": (0, 140, 235), "none": (150, 150, 150)}


def to_canonical(xy, alliance):
    """Put a path into ONE alliance's frame so runs from both sides can be compared.

    The field is 180-degree ROTATIONALLY symmetric -- station 1 sits top-left on one
    wall and bottom-right on the other -- so the same auto routine driven from the
    red side traces the rotated copy of the blue-side version. Overlaying them raw
    superimposes two opposed clusters and the team's actual route disappears into
    what looks like noise.

    Measured over all 74 teams (every one played both alliances): blue-vs-red
    occupancy similarity is 0.0034 raw, 0.087 under this rotation -- a 25x
    difference, so the correction is real and large.

    Caveat, stated because it matters: rot180 and a pure left-right mirror score
    about the same here (0.087 vs 0.105, per-team split 40/34). They differ only by
    a top-bottom flip, which this data cannot separate. rot180 is chosen from the
    field's station numbering, not from the measurement.
    """
    if alliance != "red":
        return xy
    cx = (td.FX0 + td.FX1) / 2.0
    cy = (td.FY0 + td.FY1) / 2.0
    out = np.asarray(xy, float).copy()
    out[:, 0] = 2 * cx - out[:, 0]
    out[:, 1] = 2 * cy - out[:, 1]
    return out


def to_canonical_fi(fi_pts, alliance):
    """Field-inch (N,2) path -> canonical blue frame via 180° field rotation.

    Rotation about the exact field center avoids canvas-mapper distortions.
    x' = FIELD_LEN - x, y' = FIELD_WID - y.
    """
    if alliance != "red":
        return np.asarray(fi_pts, float)
    pts = np.asarray(fi_pts, float).copy()
    pts[:, 0] = zones.FIELD_LEN - pts[:, 0]
    pts[:, 1] = zones.FIELD_WID - pts[:, 1]
    return pts


def _resample(pts, n=40):
    """Resample (N,2) path to exactly n equidistant points by arc length."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return np.tile(pts[0] if len(pts) else [0.0, 0.0], (n, 1))
    dists = np.concatenate([[0.0],
                            np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = dists[-1]
    if total < 1e-6:
        return np.tile(pts[0], (n, 1))
    t = np.linspace(0, total, n)
    xi = np.interp(t, dists, pts[:, 0])
    yi = np.interp(t, dists, pts[:, 1])
    return np.column_stack([xi, yi])


def _cluster_paths(fi_paths, rms_threshold=55.0, n_resample=40):
    """Cluster (N,2) field-inch paths by RMS positional similarity.

    Returns integer label array (1-indexed). Falls back to one cluster each if
    scipy is not installed.
    """
    n = len(fi_paths)
    if n == 1 or not _SCIPY:
        return np.ones(n, int)
    feats = np.array([_resample(p, n_resample).ravel() for p in fi_paths])
    # RMS distance: ||feat_i - feat_j|| / sqrt(n_resample) is in field inches
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(feats[i] - feats[j]) / np.sqrt(n_resample)
            D[i, j] = D[j, i] = d
    Z = linkage(squareform(D), method="average")
    return fcluster(Z, t=rms_threshold, criterion="distance")


def _fi_array_to_canvas(fi_arr):
    """(N,2) field-inch array -> canvas pixel array."""
    fi = np.asarray(fi_arr, float)
    cx = fi[:, 0] * (td.FX1 - td.FX0) / zones.FIELD_LEN + td.FX0
    cy = fi[:, 1] * (td.FY1 - td.FY0) / zones.FIELD_WID + td.FY0
    return np.column_stack([cx, cy])


def _fresh(path, data=None):
    """Is this auto_paths.json in CURRENT (1080p) crop geometry?

    Judged from the coordinates themselves, not the file's timestamp. A wall-clock
    cutoff silently inverts the moment the date rolls over -- it did, and excluded
    every file -- whereas the pixel range is a property of the data and stays true.

    720p-era paths live in a 1280x370 crop; current ones span a 1920x554 crop, so
    anything whose widest point falls short of the old crop's width is stale.
    """
    try:
        data = data if data is not None else json.load(open(path))
    except Exception:
        return False
    xs = [c[1] for o in data.get("teams", {}).values() for c in o.get("path", [])]
    return bool(xs) and max(xs) > STALE_MAX_X


def _label(team_key):
    return team_key.replace("frc", "#") if team_key and str(team_key).startswith("frc") \
        else str(team_key)


def _draw_path(img, xy, col, label=None, thick=2):
    pts = xy.astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(img, [pts], False, col, thick, cv2.LINE_AA)
    if len(pts):
        cv2.circle(img, tuple(pts[0]), 6, (255, 255, 255), -1)
        cv2.circle(img, tuple(pts[0]), 6, col, 2)              # start = dot
        cv2.circle(img, tuple(pts[-1]), 5, col, -1)            # end = filled
        if label:
            cv2.putText(img, label, (pts[0][0] + 8, pts[0][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)


def _match_blue_left(teams):
    """Guess which side is blue from mean start-x of each alliance (camera coords)."""
    bx = [o["path"][0][1] for o in teams.values() if o["alliance"] == "blue" and o["path"]]
    rx = [o["path"][0][1] for o in teams.values() if o["alliance"] == "red" and o["path"]]
    if bx and rx:
        return (sum(bx) / len(bx)) < (sum(rx) / len(rx))
    return True


def _field_in_to_canvas(path_field_in):
    """Convert [[t, fx, fy], ...] in field inches -> canvas pixel array."""
    pts = np.array([[fx, fy] for _t, fx, fy in path_field_in], float)
    cx = pts[:, 0] * (td.FX1 - td.FX0) / zones.FIELD_LEN + td.FX0
    cy = pts[:, 1] * (td.FY1 - td.FY0) / zones.FIELD_WID + td.FY0
    return np.column_stack([cx, cy])


def render_match(match, mapper):
    # Use smoothed auto paths when available (produced by smooth_trajectories.py)
    smooth_ap = os.path.join(RESULTS, match, "smoothed_auto_paths.json")
    ap = os.path.join(RESULTS, match, "auto_paths.json")
    if os.path.exists(smooth_ap):
        data = json.load(open(smooth_ap))
        teams = data.get("teams", {})
    elif os.path.exists(ap) and _fresh(ap):
        data = json.load(open(ap))
        teams = data.get("teams", {})
    else:
        return None
    if not teams:
        return None
    img = td.draw_field(blue_left=_match_blue_left(teams))
    cv2.putText(img, f"{match}  AUTO", (16, td.CH - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    counts = {}
    for key, o in sorted(teams.items()):
        col = td.BLUE_BGR if o["alliance"] == "blue" else td.RED_BGR
        cam = [(cx, cy) for t, cx, cy in o["path"]]
        if len(cam) < 2:
            continue
        src = o.get("source") or "none"
        counts[src] = counts.get(src, 0) + 1
        if o.get("coords") == "field_in":
            xy = _field_in_to_canvas(o["path"])
        else:
            xy = mapper(cam)
        # path in alliance colour; the LABEL carries how the number was determined,
        # so an unproven rank-order guess is never mistaken for a bumper read.
        _draw_path(img, xy, col, None)
        p0 = xy.astype(np.int32)[0]
        txt = f"{_label(o.get('team') or key)} [{src}]"
        cv2.putText(img, txt, (p0[0] + 9, p0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (p0[0] + 9, p0[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, SRC_COL.get(src, col), 2, cv2.LINE_AA)
    legend = "  ".join(f"{k}:{v}" for k, v in counts.items())
    cv2.putText(img, legend, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (30, 30, 30), 2)
    out = os.path.join(RESULTS, match, "topdown_auto.png")
    cv2.imwrite(out, img)
    return out


_CLUSTER_COLORS = [
    (50, 205,  50),    # lime
    (0,  140, 235),    # blue
    (0,  210, 210),    # cyan
    (200, 180,   0),   # gold
    (200,   0, 200),   # magenta
    (0,  180, 100),    # teal
    (220,  80,   0),   # orange
    (130, 100, 255),   # lavender
]


def render_team_aggregate(mapper, min_matches=2):
    """Each team's auto path clustered by routine on one top-down field.

    Paths from both alliances are rotated into the blue frame in FIELD INCH
    space (not canvas space) so the same routine driven red-side lands on top
    of the blue-side version. Distinct routines are drawn in different colours
    with ×N count labels.
    """
    per_team = defaultdict(list)   # team -> [(alliance, fi_array (N,2))]
    stale = 0

    for match_dir in sorted(glob.glob(os.path.join(RESULTS, "*"))):
        if not os.path.isdir(match_dir):
            continue
        smooth_ap = os.path.join(match_dir, "smoothed_auto_paths.json")
        raw_ap = os.path.join(match_dir, "auto_paths.json")

        if os.path.exists(smooth_ap):
            data = json.load(open(smooth_ap))
            for _st, o in data.get("teams", {}).items():
                team = o.get("team")
                if not (team and str(team).startswith("frc")
                        and len(o.get("path", [])) >= 2):
                    continue
                if o.get("coords") == "field_in":
                    fi = np.array([[fx, fy] for _t, fx, fy in o["path"]], float)
                else:
                    cam = [(cx, cy) for _t, cx, cy in o["path"]]
                    fi = zones.to_field_inches(np.asarray(mapper(cam)))
                per_team[team].append((o["alliance"], fi))
        elif os.path.exists(raw_ap) and _fresh(raw_ap):
            for _st, o in json.load(open(raw_ap)).get("teams", {}).items():
                team = o.get("team")
                if not (team and str(team).startswith("frc")
                        and len(o.get("path", [])) >= 2):
                    continue
                cam = [(cx, cy) for _t, cx, cy in o["path"]]
                fi = zones.to_field_inches(np.asarray(mapper(cam)))
                per_team[team].append((o["alliance"], fi))
        else:
            stale += 1

    if stale:
        print(f"  excluded {stale} matches with missing/stale auto_paths.json")

    os.makedirs(OUTDIR, exist_ok=True)
    kept = 0

    for team, runs in sorted(per_team.items()):
        if len(runs) < min_matches:
            continue
        kept += 1

        # Canonicalize into blue frame in field-inch space
        canonical = [to_canonical_fi(fi, al) for al, fi in runs]

        labels = _cluster_paths(canonical)
        n_clusters = int(labels.max())

        img = td.draw_field(blue_left=True)

        for cid in range(1, n_clusters + 1):
            idxs = [i for i, lb in enumerate(labels) if lb == cid]
            count = len(idxs)
            col = _CLUSTER_COLORS[(cid - 1) % len(_CLUSTER_COLORS)]
            thick = max(1, min(4, 1 + count // 2))

            for i in idxs:
                canvas_pts = _fi_array_to_canvas(canonical[i])
                _draw_path(img, canvas_pts, col, thick=thick)

            # Count label at the start of the first path in this cluster
            p0 = _fi_array_to_canvas(canonical[idxs[0]][:1]).astype(np.int32)[0]
            label_txt = f"×{count}"
            cv2.putText(img, label_txt, (p0[0] + 9, p0[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, label_txt, (p0[0] + 9, p0[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        nblue = sum(1 for al, _ in runs if al == "blue")
        nred = len(runs) - nblue
        routine_str = f"{n_clusters} routine{'s' if n_clusters != 1 else ''}"
        cv2.putText(img,
                    f"{_label(team)}  ({len(runs)} autos, {routine_str}: "
                    f"{nblue}B+{nred}R → blue frame)",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (30, 30, 30), 2)
        cv2.imwrite(os.path.join(OUTDIR, f"{_label(team).strip('#')}.png"), img)

    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", default=MATCH,
                   help="one match key; empty for all + per-team aggregate")
    p.add_argument("--min-matches", type=int, default=MIN_MATCHES)
    args = p.parse_args()

    mapper = td.load_mapper()
    if mapper is None:
        raise SystemExit("No field map -- run autofit_fieldmap.py first.")

    if args.match:
        out = render_match(args.match, mapper)
        print("wrote", out or "(no auto_paths for that match)")
        return
    n = 0
    for ap in sorted(glob.glob(os.path.join(RESULTS, "*", "auto_paths.json"))):
        m = os.path.basename(os.path.dirname(ap))
        if render_match(m, mapper):
            n += 1
    kept = render_team_aggregate(mapper, args.min_matches)
    print(f"per-match top-downs: {n}  |  per-team aggregates: {kept} -> {OUTDIR}")


if __name__ == "__main__":
    main()
