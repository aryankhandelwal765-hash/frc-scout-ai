"""
Preferred (aggregated) auto paths per team -- the "auto book".

Collects every team's auto path across all matches (from each match's
auto_paths.json), resamples them to a common length, and averages them into one
representative route. Overlays all a team's individual runs (faint) + the mean
run (bold) on the reference field frame.

Output -> match_pipeline/preferred_paths/:
  <team>.png          per-team overlay (all runs + mean)
  _all_means.png      every team's mean route on one field
  preferred.json      team -> {matches, alliance, mean_path}

  python preferred_paths.py
  python preferred_paths.py --min-matches 3
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CALIB = os.path.join(HERE, "calib.json")
OUTDIR = os.path.join(HERE, "preferred_paths")
BLUE_BGR = (255, 120, 0)
RED_BGR = (0, 80, 255)
N = 40                      # resample every path to N points


def _resample(path_xy, n=N):
    """Resample a polyline to n points evenly along its arc length."""
    pts = np.array(path_xy, float)
    if len(pts) < 2:
        return None
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] <= 1e-6:
        return None
    targets = np.linspace(0, cum[-1], n)
    x = np.interp(targets, cum, pts[:, 0])
    y = np.interp(targets, cum, pts[:, 1])
    return np.stack([x, y], 1)


def _load_all():
    """team_key -> list of (alliance, resampled_path)."""
    per_team = defaultdict(list)
    for ap_path in glob.glob(os.path.join(RESULTS, "*", "auto_paths.json")):
        data = json.load(open(ap_path))
        for _station, o in data.get("teams", {}).items():
            # auto_paths keys by STATION now, so read the team off the record --
            # the old key-is-the-team read silently collided (see auto_paths.run).
            team_key = o.get("team")
            if not (team_key and str(team_key).startswith("frc")):
                continue                       # skip unclassified robots
            rs = _resample([(cx, cy) for t, cx, cy in o["path"]])
            if rs is not None:
                per_team[team_key].append((o["alliance"], rs))
    return per_team


def _ref_image(calib):
    ref = calib.get("reference_frame")
    if ref:
        # calib.json stores this relative to match_pipeline/ so the file is portable.
        if not os.path.isabs(ref):
            ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), ref)
        if os.path.exists(ref):
            return cv2.imread(ref)
    h = calib["crop"][3] - calib["crop"][1]
    w = calib["crop"][2] - calib["crop"][0]
    return np.full((h, w, 3), 40, np.uint8)


def _team_label(k):
    return k.replace("frc", "#")


def run(min_matches=2):
    if not os.path.exists(CALIB):
        raise SystemExit("No calib.json -- run field_calibrate.py first.")
    calib = json.load(open(CALIB))
    os.makedirs(OUTDIR, exist_ok=True)
    per_team = _load_all()

    means_img = _ref_image(calib)
    out = {}
    kept = 0
    for team_key, runs in sorted(per_team.items()):
        if len(runs) < min_matches:
            continue
        kept += 1
        alliance = max(("blue", "red"), key=lambda a: sum(1 for al, _ in runs if al == a))
        col = BLUE_BGR if alliance == "blue" else RED_BGR
        stack = np.stack([r for _, r in runs], 0)     # (M, N, 2)
        mean = stack.mean(0)

        img = _ref_image(calib)
        for _, r in runs:                              # faint individual runs
            cv2.polylines(img, [r.astype(np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.polylines(img, [mean.astype(np.int32)], False, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.polylines(img, [mean.astype(np.int32)], False, col, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(mean[0].astype(int)), 6, (255, 255, 255), -1)
        cv2.putText(img, f"{_team_label(team_key)}  ({len(runs)} autos)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.imwrite(os.path.join(OUTDIR, f"{_team_label(team_key).strip('#')}.png"), img)

        cv2.polylines(means_img, [mean.astype(np.int32)], False, col, 2, cv2.LINE_AA)
        cv2.putText(means_img, _team_label(team_key), tuple(mean[0].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        out[team_key] = {"matches": len(runs), "alliance": alliance,
                         "mean_path": mean.round(1).tolist()}

    cv2.imwrite(os.path.join(OUTDIR, "_all_means.png"), means_img)
    with open(os.path.join(OUTDIR, "preferred.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"preferred_paths: {kept} teams (>= {min_matches} autos) -> {OUTDIR}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-matches", type=int, default=2)
    args = p.parse_args()
    run(min_matches=args.min_matches)


if __name__ == "__main__":
    main()
