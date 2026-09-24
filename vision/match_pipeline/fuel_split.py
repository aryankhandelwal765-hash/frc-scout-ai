"""
Per-robot FUEL estimates: TBA's exact alliance total, split by zone-gated activity.

TBA publishes how many FUEL each ALLIANCE scored, per phase and per 25-second SHIFT
(`hubScore.autoCount`, `teleopCount`, `endgameCount`, `shift1..4Count`,
`transitionCount`). It never publishes per-ROBOT counts. So the total is exact and
only the SPLIT is estimated -- which is the right way round, because it means the
alliance sums always reconcile with the official score.

THE GATE IS A RULE, NOT A GUESS. Manual G407: "Only score while in your ALLIANCE
ZONE. A ROBOT may not launch a SCORING ELEMENT into their HUB unless their BUMPERS
are partially or fully within their ALLIANCE ZONE." So a robot can only have scored
during the time it spent inside its own ALLIANCE ZONE, and time spent anywhere else
cannot have produced points. Weighting by scoring-eligible time is therefore
grounded in the rules rather than in an assumption about what robots do.

Two weights are produced because two kinds of robot both score:
  dwell    samples inside the own ALLIANCE ZONE. Captures a robot parked near the
           HUB being fed by its human player through the OUTPOST.
  entries  separate arrivals into that zone. Captures a cycler running fuel back
           from the NEUTRAL ZONE.
`WEIGHT` picks which drives the split; "blend" averages the two normalised shares.

WHY THE ROWS DO NOT SUM TO THE TBA TOTAL, and why that is correct. A robot's share
is its weight over the weight of ALL THREE robots, including any that could not be
identified. So each named robot's estimate is unbiased on its own, and the named
rows deliberately sum to LESS than the alliance total -- the missing amount is what
the unidentified robot(s) scored. Inflating the named robots to close that gap would
make the sum look tidy and every individual number wrong. `share_covered` reports
how much of the alliance weight was attributable.

WHAT CANNOT BE CHECKED. TBA has no per-robot ball count, so the split has no ground
truth anywhere. The alliance TOTAL is exact; the per-robot numbers are an informed
apportionment and nothing more.

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
import zones

RESULTS = os.path.join(HERE, "results")
TRUTH = os.path.join(HERE, "truth")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
WEIGHT = "blend"         # "dwell" | "entries" | "blend"
MIN_COVER = 0.5          # warn when identified robots hold less than this share
OUT = os.path.join(HERE, "fuel_split.csv")
ONLY = []                # match keys; [] = every current-geometry match
# ===========================================================================

# (TBA hubScore key, seconds from AUTO start). Manual timing: AUTO 20s, then
# TRANSITION 10s, SHIFT 1-4 at 25s each, END GAME 30s.
PHASES = [
    ("autoCount",       0.0,   20.0),
    ("transitionCount", 20.0,  30.0),
    ("shift1Count",     30.0,  55.0),
    ("shift2Count",     55.0,  80.0),
    ("shift3Count",     80.0,  105.0),
    ("shift4Count",     105.0, 130.0),
    ("endgameCount",    130.0, 160.0),
]


def _weights(match, mapper, rosters=None):
    """(alliance, team|None) -> {phase: (dwell, entries)} of scoring-eligible time."""
    apf = os.path.join(RESULTS, match, "auto_paths.json")
    if not os.path.exists(apf) or not tp._fresh(apf):
        return {}, 60.0
    tracks = ap._load_trajectory(match)
    if not tracks:
        return {}, 60.0
    fps = 60.0
    tj = os.path.join(RESULTS, match, "tracks.json")
    if os.path.exists(tj):
        fps = json.load(open(tj)).get("fps", 60.0)
    start = ap._detected_start(match)
    if start is None:
        start = int(ap.AUTO_START_SEC * fps)
    start = max(0, start)

    al_of = {t: ap._voted_alliance(match, t, r) for t, r in tracks.items()}
    last = max(r[0] for rows in tracks.values() for r in rows)
    chain_list = stitch.chains(match, tracks, al_of, fps=fps, window=(0, last))
    locked, _ = stitch.lock_identity(chain_list, ap._read_identity(match), tracks)

    teams = json.load(open(apf)).get("teams", {})
    blue_left = tp._match_blue_left(teams)

    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for ch in chain_list:
        al = al_of.get(ch[0])
        if al not in ("blue", "red"):
            continue
        team = next((locked[t] for t in ch if t in locked), None)
        # A read can attach to a track whose alliance was called wrong, which puts a
        # team on the alliance it never played for -- qm27 had 4698 landing on BOTH.
        # TBA's roster is authoritative, so drop the identity and keep the weight as
        # unattributed rather than crediting the wrong alliance.
        if team and rosters and team not in rosters.get(al, ()):
            team = None
        pts = sorted(r for tid in ch for r in tracks.get(tid, []))
        if len(pts) < 10:
            continue
        fi = zones.to_field_inches(mapper([(r[3], r[4]) for r in pts]))
        frames = np.array([r[0] for r in pts])
        secs = (frames - start) / fps
        own_left = (al == "blue") == blue_left
        # G407: scoring-eligible means inside the robot's OWN alliance zone
        edge = zones.BAND_NEAR
        inzone = (fi[:, 0] <= edge) if own_left else (fi[:, 0] >= zones.FIELD_LEN - edge)
        for key, t0, t1 in PHASES:
            m = (secs >= t0) & (secs < t1) & inzone
            if not m.any():
                continue
            rec = out[(al, team)][key]
            rec[0] += int(m.sum())
            # an "entry" is a rising edge into the zone within this phase
            e = np.diff(np.concatenate([[False], m])).astype(bool) & m
            rec[1] += int(e.sum())
    return out, fps


def run_match(match, mapper):
    truth_p = os.path.join(TRUTH, f"{match}.json")
    if not os.path.exists(truth_p):
        return []
    truth = json.load(open(truth_p))
    sb = truth.get("score_breakdown") or {}
    rosters = {a: {t.replace("frc", "") for t in (truth.get(f"{a}_teams") or [])}
               for a in ("blue", "red")}
    w, _fps = _weights(match, mapper, rosters)
    if not w:
        return []

    rows = []
    for al in ("blue", "red"):
        hub = ((sb.get(al) or {}).get("hubScore") or {})
        if not hub:
            continue
        members = {k[1]: v for k, v in w.items() if k[0] == al}
        if not members:
            continue
        per_team = defaultdict(dict)
        cover = {}
        for key, _t0, _t1 in PHASES:
            tot = hub.get(key)
            if tot in (None, ""):
                continue
            d = {t: v[key][0] for t, v in members.items() if key in v}
            e = {t: v[key][1] for t, v in members.items() if key in v}
            sd, se = sum(d.values()), sum(e.values())
            if sd <= 0 and se <= 0:
                continue
            # share of the alliance's weight that belongs to an IDENTIFIED robot
            known = sum(v for t, v in d.items() if t) / sd if sd else 0.0
            cover[key] = known
            for t in members:
                if not t:
                    continue          # unidentified: weight counts, but no row
                wd = d.get(t, 0) / sd if sd else 0.0
                we = e.get(t, 0) / se if se else 0.0
                share = {"dwell": wd, "entries": we,
                         "blend": (wd + we) / 2.0}[WEIGHT]
                per_team[t][key] = share * float(tot)
        for t, ph in per_team.items():
            auto = ph.get("autoCount", 0.0)
            tele = sum(ph.get(k, 0.0) for k in
                       ("transitionCount", "shift1Count", "shift2Count",
                        "shift3Count", "shift4Count"))
            end = ph.get("endgameCount", 0.0)
            rows.append({
                "match": match, "team": t, "alliance": al,
                "auto_fuel": round(auto, 1),
                "teleop_fuel": round(tele, 1),
                "endgame_fuel": round(end, 1),
                "total_fuel": round(auto + tele + end, 1),
                **{k.replace("Count", "_fuel"): round(ph.get(k, 0.0), 1)
                   for k, _a, _b in PHASES},
                "share_covered": round(float(np.mean(list(cover.values()) or [0])), 2),
                "weight_mode": WEIGHT,
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    mapper = td.load_mapper()
    if mapper is None:
        raise SystemExit("No field map -- run multi_video_apriltag_calib.py first.")

    matches = ONLY or sorted(
        os.path.basename(os.path.dirname(x))
        for x in glob.glob(os.path.join(RESULTS, "*", "auto_paths.json")))
    allrows = []
    for i, m in enumerate(matches, 1):
        try:
            allrows += run_match(m, mapper)
        except Exception as e:
            print(f"  {m}: {e.__class__.__name__}: {e}")
        if i % 25 == 0:
            print(f"  ... {i}/{len(matches)}")
    if not allrows:
        raise SystemExit("no rows produced")

    cols = list(allrows[0].keys())
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(allrows)

    cov = np.array([r["share_covered"] for r in allrows])
    tf = np.array([r["total_fuel"] for r in allrows])
    print(f"\n{len(allrows)} team-match fuel estimates -> {a.out}")
    print(f"  weight mode: {WEIGHT}")
    print(f"  total_fuel per robot: median {np.median(tf):.0f}  "
          f"p10 {np.percentile(tf, 10):.0f}  p90 {np.percentile(tf, 90):.0f}")
    print(f"  share_covered: median {np.median(cov):.2f}  "
          f"below {MIN_COVER}: {(cov < MIN_COVER).sum()} rows "
          f"({100.0 * (cov < MIN_COVER).mean():.0f}%)")
    print("\n  The alliance TOTAL is TBA's and exact. The per-robot split has no ground")
    print("  truth -- TBA publishes no per-robot ball count.")
    print("  Named rows sum to LESS than the alliance total ON PURPOSE: each robot's")
    print("  share is over ALL three robots, so an unidentified teammate's fuel is")
    print("  simply absent rather than redistributed. Individual numbers stay unbiased;")
    print("  the alliance sum is only complete where share_covered is near 1.")


if __name__ == "__main__":
    main()
