"""
Per-team TELEOP paths and the zone facts derived from them.

This closes the last two vision columns in the scouting table -- tele_bump and
tele_trench -- and adds the counts a hand scout cannot produce. A human ticks a box
for "used the trench"; this reports how many separate times the robot went through
one, and which zones it actually spent teleop in.

The path is the whole-match stitched chain restricted to teleop, so a robot keeps
one continuous path across the ByteTrack id changes that happen constantly during
open play. Identity comes from a bumper read anywhere on that chain.

Teleop starts at the detected AUTO start + 20s (manual: AUTO is 20 seconds).

Output -> teleop_facts.json   { match: { team: {...} } }
Precomputed rather than done inline by scout_table.py, because stitching 112
matches is the expensive part and the table gets rebuilt often.

PyCharm: click Run.
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

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

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
AUTO_SEC = 20.0  # manual: AUTO is 20s; teleop analysis begins after it
MIN_SAMPLES = 30  # a chain needs this many teleop samples to be usable
OUT = os.path.join(HERE, "teleop_facts.json")
ONLY = []  # match keys; [] = every current-geometry match
# ===========================================================================


def run_match(match, mapper=None):
    """team -> teleop zone facts for one match."""
    apf = os.path.join(RESULTS, match, "auto_paths.json")
    if not os.path.exists(apf) or not tp._fresh(apf):
        return {}
    tracks = ap._load_trajectory(match)
    if not tracks:
        return {}
    mapper = mapper or td.load_mapper()
    if mapper is None:
        return {}

    fps = 60.0
    tj = os.path.join(RESULTS, match, "tracks.json")
    if os.path.exists(tj):
        fps = json.load(open(tj)).get("fps", 60.0)
    start = ap._detected_start(match)
    if start is None:
        start = int(ap.AUTO_START_SEC * fps)
    tstart = max(0, start) + int(AUTO_SEC * fps)

    al_of = {t: ap._voted_alliance(match, t, r) for t, r in tracks.items()}
    last = max(r[0] for rows in tracks.values() for r in rows)
    chain_list = stitch.chains(match, tracks, al_of, fps=fps, window=(0, last))
    locked, _ = stitch.lock_identity(chain_list, ap._read_identity(match), tracks)

    # a team's teleop is spread over several chains, so accumulate per team
    per = defaultdict(lambda: {"samples": 0, "seq": [], "zone_time": Counter()})
    for ch in chain_list:
        team = next((locked[t] for t in ch if t in locked), None)
        if not team:
            continue
        pts = sorted(r for tid in ch for r in tracks.get(tid, []) if r[0] >= tstart)
        if len(pts) < MIN_SAMPLES:
            continue
        fi = zones.to_field_inches(mapper([(r[3], r[4]) for r in pts]))
        rec = per[team]
        rec["samples"] += len(fi)
        rec["seq"] += zones.zone_sequence(fi)
        for x, y in fi:
            zn = zones.zone_at(x, y)
            if zn:
                rec["zone_time"][zn] += 1

    out = {}
    for team, rec in per.items():
        seq = rec["seq"]
        # a "visit" is an ENTRY into that kind of zone, i.e. a hit in the collapsed
        # sequence -- not a sample count, which would just measure dwell time.
        bumps = sum(1 for z in seq if "Bump" in z)
        trenches = sum(1 for z in seq if "Trench" in z)
        hubs = sum(1 for z in seq if "Hub" in z)
        tot = sum(rec["zone_time"].values()) or 1
        top = rec["zone_time"].most_common(3)
        out[team] = {
            "tele_bump": int(bumps > 0),
            "tele_trench": int(trenches > 0),
            "tele_bump_visits": bumps,
            "tele_trench_visits": trenches,
            "tele_hub_visits": hubs,
            "tele_samples": rec["samples"],
            "tele_top_zones": "; ".join(f"{z} {100.0 * n / tot:.0f}%" for z, n in top),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    matches = ONLY or sorted(
        os.path.basename(os.path.dirname(x))
        for x in glob.glob(os.path.join(RESULTS, "*", "auto_paths.json"))
    )
    mapper = td.load_mapper()
    if mapper is None:
        raise SystemExit("No field map -- run autofit_fieldmap.py first.")

    allout, nm = {}, 0
    for i, m in enumerate(matches, 1):
        try:
            r = run_match(m, mapper)
        except Exception as e:
            print(f"  {m}: {e.__class__.__name__}: {e}")
            continue
        if r:
            allout[m] = r
            nm += 1
        if i % 25 == 0:
            print(f"  ... {i}/{len(matches)}")
    json.dump(allout, open(a.out, "w"), indent=1)

    rows = [v for m in allout.values() for v in m.values()]
    if not rows:
        raise SystemExit("no teleop facts produced")
    nb = sum(r["tele_bump"] for r in rows)
    nt = sum(r["tele_trench"] for r in rows)
    print(f"\n{len(rows)} team-matches with teleop paths across {nm} matches -> {a.out}")
    print(
        f"  tele_bump   used: {nb}/{len(rows)} = {100.0 * nb / len(rows):.0f}%"
        f"   (hand sheet said 59.9%)"
    )
    print(
        f"  tele_trench used: {nt}/{len(rows)} = {100.0 * nt / len(rows):.0f}%"
        f"   (hand sheet said 56.9%)"
    )
    for k in ("tele_bump_visits", "tele_trench_visits", "tele_hub_visits"):
        v = np.array([r[k] for r in rows])
        print(
            f"  {k:20s} median {np.median(v):4.0f}  p90 {np.percentile(v, 90):4.0f}"
            f"  max {v.max():4.0f}"
        )


if __name__ == "__main__":
    main()
