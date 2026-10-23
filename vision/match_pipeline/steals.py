"""
STEALS: a robot crosses into the opponent's half, grabs FUEL, and runs it home.

A steal is a round trip, and a round trip is pure motion -- which is the one thing
this pipeline measures well. We never see the ball, so "took FUEL" is inferred from
the excursion itself, not observed. That inference is the metric's main weakness and
it is stated in the output rather than hidden.

  excursion   the robot's ground point crosses the CENTER LINE into the far half
              and later comes back. Teleop only: the manual makes completely
              crossing the CENTER LINE during AUTO a MAJOR FOUL, so any auto-period
              "steal" is a tracking error, not a play.
  steal       an excursion that makes it back to the robot's OWN ALLIANCE ZONE
              (the 158.6in-deep band in front of its own wall, per manual 5.3).
  successful  a steal whose return leg is quick and uninterrupted -- see below.

Two readings of "back within 4.5 seconds" are both reported, because the phrasing is
genuinely ambiguous and they measure different things:

  return_time   deepest point in enemy territory -> arrival in own ALLIANCE ZONE.
                This is the clean-getaway reading: once it turns around, how fast
                does it get home.
  away_time     crossing the CENTER LINE out -> arrival in own ALLIANCE ZONE.
                Total round-trip exposure, which includes time spent collecting.

Track fragmentation matters here more than anywhere else: crossing the field is
exactly when ByteTrack loses a robot, so this stitches chains over the WHOLE match
rather than the auto window, then locks identity from any bumper read on the chain.

PyCharm: click Run.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
import auto_paths as ap
import stitch
import topdown as td
import topdown_paths as tp

RESULTS = os.path.join(HERE, "results")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
RETURN_SEC = 4.5  # a steal is "successful" if the return leg is <= this
ALLIANCE_DEPTH_IN = 158.6  # manual 5.3: ALLIANCE ZONE depth from the ALLIANCE WALL
FIELD_LEN_IN = 651.2  # manual 5.2
MIN_DEPTH_IN = 36.0  # must get this far past the CENTER LINE to count as a
# real excursion, not tracking jitter across the line
MIN_SAMPLES = 4  # samples required on the far side
AUTO_SEC = 20.0  # manual: AUTO is 20s. Teleop analysis starts after it.
OUT = os.path.join(HERE, "steals.csv")
ONLY = []  # match keys; [] = every match with current-geometry data
# ===========================================================================


def _teleop_start(match, fps):
    """First frame of teleop = detected auto start + the 20s auto period."""
    sf = ap._detected_start(match)
    if sf is None:
        sf = int(ap.AUTO_START_SEC * fps)
    return max(0, sf) + int(AUTO_SEC * fps)


def excursions(xy, frames, own_left, center, alliance_edge, fps, px_per_in):
    """Find far-half round trips for one robot. Returns list of dicts."""
    # signed distance into ENEMY territory, in inches
    depth = ((xy[:, 0] - center) if own_left else (center - xy[:, 0])) / px_per_in
    # distance home: how deep into its OWN alliance zone the robot is
    home = ((alliance_edge - xy[:, 0]) if own_left else (xy[:, 0] - alliance_edge)) / px_per_in

    out = []
    i, n = 0, len(depth)
    while i < n:
        if depth[i] <= 0:
            i += 1
            continue
        j = i
        while j < n and depth[j] > 0:
            j += 1
        seg = slice(i, j)
        if (j - i) >= MIN_SAMPLES and depth[seg].max() >= MIN_DEPTH_IN:
            k = i + int(np.argmax(depth[seg]))  # deepest point
            # first sample AFTER the excursion where the robot is home
            home_idx = None
            for t in range(j, n):
                if home[t] >= 0:
                    home_idx = t
                    break
            if home_idx is not None:
                out.append(
                    {
                        "out_frame": int(frames[i]),
                        "deep_frame": int(frames[k]),
                        "home_frame": int(frames[home_idx]),
                        "max_depth_in": round(float(depth[seg].max()), 1),
                        "return_sec": round(float(frames[home_idx] - frames[k]) / fps, 2),
                        "away_sec": round(float(frames[home_idx] - frames[i]) / fps, 2),
                    }
                )
        i = j
    return out


def run_match(match):
    tpath = os.path.join(RESULTS, match, "trajectory.csv")
    if not os.path.exists(tpath):
        return []
    apf = os.path.join(RESULTS, match, "auto_paths.json")
    if not os.path.exists(apf) or not tp._fresh(apf):
        return []  # pre-1080p geometry: coordinates are not comparable

    mapper = td.load_mapper()
    tracks = ap._load_trajectory(match)
    if not tracks:
        return []
    fps = 60.0
    tj = os.path.join(RESULTS, match, "tracks.json")
    if os.path.exists(tj):
        fps = json.load(open(tj)).get("fps", 60.0)

    al_of = {tid: ap._voted_alliance(match, tid, rows) for tid, rows in tracks.items()}
    # whole-match chains: a robot crossing the field is exactly when its id breaks
    last = max(r[0] for rows in tracks.values() for r in rows)
    read_id = ap._read_identity(match)
    chain_list = stitch.whole_match_chains(match, tracks, al_of, read_id, fps=fps, window=(0, last))
    locked, _ = stitch.lock_identity(chain_list, read_id, tracks)

    teams = json.load(open(apf)).get("teams", {})
    blue_left = tp._match_blue_left(teams)
    px_per_in = (td.FX1 - td.FX0) / FIELD_LEN_IN
    center = (td.FX0 + td.FX1) / 2.0
    depth_px = ALLIANCE_DEPTH_IN * px_per_in
    tstart = _teleop_start(match, fps)

    rows = []
    for ch in chain_list:
        alliance = al_of.get(ch[0])
        if alliance not in ("blue", "red"):
            continue
        pts = sorted(r for tid in ch for r in tracks.get(tid, []) if r[0] >= tstart)
        if len(pts) < 30:
            continue
        xy = mapper([(r[3], r[4]) for r in pts])
        frames = np.array([r[0] for r in pts])
        own_left = (alliance == "blue") == blue_left
        alliance_edge = (td.FX0 + depth_px) if own_left else (td.FX1 - depth_px)
        team = next((locked[t] for t in ch if t in locked), None)
        for e in excursions(xy, frames, own_left, center, alliance_edge, fps, px_per_in):
            e.update(
                {
                    "match": match,
                    "alliance": alliance,
                    "team": team or "",
                    "tids": " ".join(map(str, ch)),
                    "identified": "read" if team else "unknown",
                    "successful": int(e["return_sec"] <= RETURN_SEC),
                }
            )
            rows.append(e)
    return _merge_overlaps(rows)


def _merge_overlaps(rows):
    """One team is one robot: two excursions of the same team whose out->home spans overlap
    are the same trip seen through two chains (or one chain mislabelled), so keep the deeper.
    Unidentified rows have no team to compare, so they pass through."""
    keep, by_team = [], defaultdict(list)
    for r in rows:
        (by_team[r["team"]] if r["team"] else keep).append(r)
    for rs in by_team.values():
        rs.sort(key=lambda r: r["out_frame"])
        cur = rs[0]
        for r in rs[1:]:
            if r["out_frame"] < cur["home_frame"]:
                if r["max_depth_in"] > cur["max_depth_in"]:
                    cur = {**r, "out_frame": min(r["out_frame"], cur["out_frame"])}
                continue
            keep.append(cur)
            cur = r
        keep.append(cur)
    return sorted(keep, key=lambda r: r["out_frame"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    matches = ONLY or sorted(
        os.path.basename(os.path.dirname(x))
        for x in glob.glob(os.path.join(RESULTS, "*", "trajectory.csv"))
    )
    allrows = []
    for i, m in enumerate(matches, 1):
        try:
            allrows += run_match(m)
        except Exception as e:
            print(f"  {m}: {e.__class__.__name__}: {e}")
        if i % 25 == 0:
            print(f"  ... {i}/{len(matches)}")

    if not allrows:
        raise SystemExit("no excursions found")
    cols = [
        "match",
        "team",
        "alliance",
        "identified",
        "tids",
        "out_frame",
        "deep_frame",
        "home_frame",
        "max_depth_in",
        "return_sec",
        "away_sec",
        "successful",
    ]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)

    ret = np.array([r["return_sec"] for r in allrows])
    away = np.array([r["away_sec"] for r in allrows])
    ident = sum(1 for r in allrows if r["team"])
    print(
        f"\n{len(allrows)} excursions across {len({r['match'] for r in allrows})} "
        f"matches -> {a.out}"
    )
    print(f"  attributed to a READ team: {ident} ({ident / len(allrows) * 100:.0f}%)")
    print(
        f"  return leg (turnaround -> home): median {np.median(ret):.1f}s  "
        f"p25 {np.percentile(ret, 25):.1f}s  p75 {np.percentile(ret, 75):.1f}s"
    )
    print(f"  away time  (out -> home):        median {np.median(away):.1f}s")
    print(
        f"  successful by return_sec <= {RETURN_SEC}: "
        f"{(ret <= RETURN_SEC).sum()} ({(ret <= RETURN_SEC).mean() * 100:.0f}%)"
    )
    print(
        f"  successful by away_sec   <= {RETURN_SEC}: "
        f"{(away <= RETURN_SEC).sum()} ({(away <= RETURN_SEC).mean() * 100:.0f}%)"
    )

    per = defaultdict(lambda: [0, 0])
    for r in allrows:
        if not r["team"]:
            continue
        per[r["team"]][0] += 1
        per[r["team"]][1] += r["successful"]
    top = sorted(per.items(), key=lambda kv: -kv[1][0])[:10]
    print("\n  most active stealers (steals / successful):")
    for t, (n, s) in top:
        print(f"     {t:7s} {n:3d} / {s:3d}")


if __name__ == "__main__":
    main()
