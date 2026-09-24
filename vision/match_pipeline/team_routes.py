"""
Per-team AUTO ROUTES: cluster a team's autos by where they started, then draw the
MEAN path of each cluster.

Why not one mean per team: plenty of teams run a different routine depending on
which side of their wall they line up on. Averaging those together produces a path
the robot never drove -- it splits the difference between two real routes and
represents neither. So cluster the runs by start position first and average within
each cluster.

Everything is drawn in ONE alliance frame: red-side runs are rotated 180 degrees
(the field is rotationally symmetric), so the same routine lands in the same place
whichever side it was run from. See topdown_paths.to_canonical.

Output -> team_routes/<team>.png       one page per team
       -> team_routes/_index.png       every team's dominant route on one field

PyCharm: set TEAM (or leave "" for all), hit Run.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
import topdown as td
import topdown_paths as tp
import zones

RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(HERE, "team_routes")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
TEAM = ""             # e.g. "frc2056"; "" = every team
MIN_RUNS = 3          # a team needs this many usable autos to get a page
SAMPLES = 60          # points each path is resampled to before averaging
MAX_CLUSTERS = 3      # most start-groups to split a team into
SPLIT_DIST = 170      # cluster centres closer than this (top-down px) are one group
ONTIME_ONLY = True    # only use runs whose track exists at the auto-window start.
                      # A run picked up mid-auto has no meaningful start position --
                      # measured: only 58% of paths begin within 0.5s of the window,
                      # and the rest begin up to 8s late.
ONTIME_FRAMES = 30    # "at the start" = within this many frames of the window start
DELAY_SPLIT_S = 1.5   # gap between fast and slow sub-groups to count as "delayed" variant
DELAY_MIN_EACH = 2    # both sub-groups need at least this many runs before we split
# ===========================================================================


def _resample(xy, n=SAMPLES):
    """Resample a path to n points evenly in TIME (all autos share one window, so
    time-alignment keeps 'where was it at t' comparable across runs)."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return None
    src = np.linspace(0.0, 1.0, len(xy))
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, xy[:, 0]),
                     np.interp(dst, src, xy[:, 1])], 1)


def _start_delay_s(path_fi_t, threshold_in=30.0):
    """Seconds from first detection to first move > threshold_in from start position."""
    if len(path_fi_t) < 2:
        return 0.0
    t0, x0, y0 = path_fi_t[0]
    for t, x, y in path_fi_t[1:]:
        if ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 > threshold_in:
            return round(t - t0, 2)
    return round(path_fi_t[-1][0] - t0, 2)


def load_runs(mapper):
    """team -> list of run dicts: {alliance, rs, fi_can, zone_seq, delay_s, match, station}

    rs      -- (SAMPLES, 2) canvas-px path resampled for rendering/clustering
    fi_can  -- (N, 2) field-inch canonical path for zone sequence extraction
    """
    per = defaultdict(list)
    skipped_late = 0

    # Prefer smoothed_auto_paths.json per match when available
    match_dirs = sorted(
        d for d in glob.glob(os.path.join(RESULTS, "*")) if os.path.isdir(d))

    for match_dir in match_dirs:
        match = os.path.basename(match_dir)
        smooth_ap = os.path.join(match_dir, "smoothed_auto_paths.json")
        raw_ap = os.path.join(match_dir, "auto_paths.json")

        if os.path.exists(smooth_ap):
            try:
                d = json.load(open(smooth_ap))
            except Exception:
                continue
            use_smooth = True
        elif os.path.exists(raw_ap) and tp._fresh(raw_ap):
            try:
                d = json.load(open(raw_ap))
            except Exception:
                continue
            use_smooth = False
        else:
            continue

        dbg = d.get("debug", {})
        sf, fps = dbg.get("start_frame", 0), dbg.get("fps", 60.0)

        for station, o in d.get("teams", {}).items():
            team = o.get("team")
            path = o.get("path") or []
            if not team or len(path) < 5:
                continue
            if ONTIME_ONLY and (path[0][0] * fps - sf) > ONTIME_FRAMES:
                skipped_late += 1
                continue

            alliance = o.get("alliance", "blue")

            # Canvas-pixel path for rendering (existing logic)
            if use_smooth and o.get("coords") == "field_in":
                canvas = np.array([[
                    fx * (td.FX1 - td.FX0) / zones.FIELD_LEN + td.FX0,
                    fy * (td.FY1 - td.FY0) / zones.FIELD_WID + td.FY0,
                ] for _t, fx, fy in path], float)
                fi = np.array([[fx, fy] for _t, fx, fy in path], float)
                pts_fi_t = [(row[0], row[1], row[2]) for row in path]
            else:
                canvas = np.asarray(mapper([(c[1], c[2]) for c in path]), float)
                fi = zones.to_field_inches(canvas)
                pts_fi_t = [(path[i][0], fi[i, 0], fi[i, 1]) for i in range(len(fi))]

            xy = tp.to_canonical(canvas, alliance)
            rs = _resample(xy)
            if rs is None:
                continue

            fi_can = tp.to_canonical_fi(fi, alliance)
            zone_seq = zones.zone_sequence(fi_can.tolist(), min_dwell=3)
            delay = _start_delay_s(pts_fi_t)

            per[team].append({
                "alliance": alliance,
                "rs": rs,
                "fi_can": fi_can,
                "zone_seq": zone_seq,
                "delay_s": delay,
                "match": match,
                "station": station,
            })

    if skipped_late:
        print(f"  skipped {skipped_late} runs that began mid-auto (no usable start)")
    return per


def cluster_starts(runs):
    """Group runs by start position, then sub-split by delay. Returns groups of indices.

    Spatial: Lloyd's k-means on the 2-D start point, k=1..MAX_CLUSTERS, keeping the
    largest k whose centres all stay >= SPLIT_DIST apart.
    Temporal: within each spatial group, if fast-starters and slow-starters are clearly
    bimodal (gap > DELAY_SPLIT_S and each side has >= DELAY_MIN_EACH runs), split again.

    Returns [[index, ...], ...], sorted biggest-first.
    """
    starts = np.array([r["rs"][0] for r in runs])

    # --- Spatial clustering ---
    best = [list(range(len(runs)))]
    for k in range(2, min(MAX_CLUSTERS, len(runs)) + 1):
        idx = np.linspace(0, len(starts) - 1, k).astype(int)
        cent = starts[np.argsort(starts[:, 1])][idx].astype(float)
        lab = None
        for _ in range(40):
            d = ((starts[:, None, :] - cent[None, :, :]) ** 2).sum(2)
            new = d.argmin(1)
            if lab is not None and (new == lab).all():
                break
            lab = new
            for j in range(k):
                if (lab == j).any():
                    cent[j] = starts[lab == j].mean(0)
        if lab is None or len(set(lab.tolist())) < k:
            continue
        sep = min(np.linalg.norm(cent[a] - cent[b])
                  for a in range(k) for b in range(a + 1, k))
        if sep < SPLIT_DIST or min((lab == j).sum() for j in range(k)) < 2:
            continue
        best = [np.where(lab == j)[0].tolist() for j in range(k)]

    # --- Delay sub-split within each spatial group ---
    final = []
    for group in best:
        delays = np.array([runs[i]["delay_s"] for i in group])
        # Find the largest gap in sorted delays
        sorted_d = np.sort(delays)
        if len(sorted_d) < DELAY_MIN_EACH * 2:
            final.append(group)
            continue
        gaps = np.diff(sorted_d)
        split_pos = int(np.argmax(gaps))
        gap_size = gaps[split_pos]
        if gap_size < DELAY_SPLIT_S:
            final.append(group)
            continue
        threshold = (sorted_d[split_pos] + sorted_d[split_pos + 1]) / 2.0
        fast = [i for i in group if runs[i]["delay_s"] <= threshold]
        slow = [i for i in group if runs[i]["delay_s"] > threshold]
        if len(fast) >= DELAY_MIN_EACH and len(slow) >= DELAY_MIN_EACH:
            final.extend([fast, slow])
        else:
            final.append(group)

    return sorted(final, key=len, reverse=True)


def _modal_zone_seq(runs, indices):
    """Most common zone sequence among the given runs (plurality vote per step)."""
    seqs = [runs[i]["zone_seq"] for i in indices if runs[i]["zone_seq"]]
    if not seqs:
        return []
    max_len = max(len(s) for s in seqs)
    modal = []
    for pos in range(max_len):
        col = [s[pos] for s in seqs if pos < len(s)]
        if col:
            modal.append(max(set(col), key=col.count))
    return modal


GROUP_COL = [(40, 175, 60), (200, 120, 20), (40, 90, 220)]


def render_team(team, runs, groups=None, out=None):
    if groups is None:
        groups = cluster_starts(runs)
    img = td.draw_field(blue_left=True)
    nblue = sum(1 for r in runs if r["alliance"] == "blue")

    for gi, g in enumerate(groups):
        col = GROUP_COL[gi % len(GROUP_COL)]
        for i in g:
            pts = runs[i]["rs"].astype(np.int32)
            cv2.polylines(img, [pts], False, (205, 205, 205), 1, cv2.LINE_AA)
        mean = np.stack([runs[i]["rs"] for i in g]).mean(0)
        dev = np.stack([np.linalg.norm(runs[i]["rs"] - mean, axis=1) for i in g]).mean(0)
        for p, r in zip(mean[::4].astype(np.int32), dev[::4]):
            cv2.circle(img, tuple(p), max(2, int(r / 3)), col, 1, cv2.LINE_AA)
        mp = mean.astype(np.int32)
        cv2.polylines(img, [mp], False, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.polylines(img, [mp], False, col, 4, cv2.LINE_AA)
        cv2.circle(img, tuple(mp[0]), 11, (255, 255, 255), -1)
        cv2.circle(img, tuple(mp[0]), 11, col, 3)
        cv2.arrowedLine(img, tuple(mp[-8]), tuple(mp[-1]), col, 4, cv2.LINE_AA,
                        tipLength=0.6)
        mean_delay = np.mean([runs[i]["delay_s"] for i in g])
        delay_tag = f" (delayed ~{mean_delay:.1f}s)" if mean_delay > DELAY_SPLIT_S else ""
        lbl = f"start {chr(65 + gi)}  n={len(g)}{delay_tag}"
        cv2.putText(img, lbl, (mp[0][0] + 16, mp[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, lbl, (mp[0][0] + 16, mp[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)

    title = (f"{tp._label(team)}   {len(runs)} autos "
             f"({nblue} blue + {len(runs) - nblue} red rotated in)   "
             f"{len(groups)} start group{'s' if len(groups) > 1 else ''}")
    cv2.putText(img, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (30, 30, 30), 2)
    cv2.putText(img, "thick = mean route   rings = spread   grey = individual autos",
                (16, td.CH - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (110, 110, 110), 1)
    os.makedirs(OUTDIR, exist_ok=True)
    out = out or os.path.join(OUTDIR, f"{tp._label(team).strip('#')}.png")
    cv2.imwrite(out, img)
    return out, len(groups)


def export_json(per_team, out_path):
    """Write team_routes.json with cluster metadata for use by auto_book.py."""
    result = {}
    for team, runs in per_team.items():
        if len(runs) < MIN_RUNS:
            continue
        groups = cluster_starts(runs)
        clusters = []
        for gi, g in enumerate(groups):
            mean_delay = float(np.mean([runs[i]["delay_s"] for i in g]))
            delayed = mean_delay > DELAY_SPLIT_S
            modal_seq = _modal_zone_seq(runs, g)
            matches = [runs[i]["match"] for i in g]
            clusters.append({
                "id": gi,
                "n_runs": len(g),
                "delayed": delayed,
                "mean_delay_s": round(mean_delay, 2),
                "zone_sequence": modal_seq,
                "matches": matches,
            })
        result[team] = {"clusters": clusters, "n_total": len(runs)}
    json.dump(result, open(out_path, "w"), indent=2)
    return len(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--team", default=TEAM, help="single team key; default = all")
    p.add_argument("--json-only", action="store_true",
                   help="export team_routes.json only, skip PNG rendering")
    a = p.parse_args()

    mapper = td.load_mapper()
    if mapper is None:
        raise SystemExit("No field map -- run multi_video_apriltag_calib.py first.")

    per = load_runs(mapper)
    json_path = os.path.join(HERE, "team_routes.json")
    n_exported = export_json(per, json_path)
    print(f"Exported {n_exported} teams -> {json_path}")

    if a.json_only:
        return

    teams = [a.team] if a.team else sorted(per)
    n = multi = 0
    for t in teams:
        runs = per.get(t, [])
        if len(runs) < MIN_RUNS:
            continue
        _out, k = render_team(t, runs)
        n += 1
        multi += k > 1
    print(f"team route pages: {n} -> {OUTDIR}")
    print(f"  {multi} teams show more than one distinct start group")


if __name__ == "__main__":
    main()
