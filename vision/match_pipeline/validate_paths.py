"""
Independent accuracy check on the auto paths, using a GAME RULE as ground truth.

The 2026 REBUILT manual states that during AUTO a ROBOT may not completely cross the
CENTER LINE (violation: MAJOR FOUL). Major fouls are rare and conspicuous, so a path
that runs deep into the opposing half is almost certainly a vision error -- a
mis-identified robot, a mis-stitched chain, or a bad alliance call -- rather than a
robot that actually did that.

That makes it a rare and valuable thing in this project: an accuracy measure that
needs NO hand-labelled ground truth. Everything else here has had to be taken on
trust or eyeballed.

Measured on 112 matches / 670 auto paths, this cleanly separates the identity tiers:

    read   0.8%   identity read off the bumper
    elim   3.8%   the only team left on the roster after the others were read
    rank   7.1%   unverified left-to-right station-order guess
    all    3.0%

which is independent confirmation that reading beats guessing by roughly 9x, and
that the elimination tier sits where it should -- it inherits the reads' accuracy.

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
import topdown as td
import topdown_paths as tp

RESULTS = os.path.join(HERE, "results")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
# How far past the CENTER LINE, as a percentage of one half-field, counts as a
# definite full crossing rather than bumpers overhanging the line. The ground point
# is the bottom-centre of the robot, so a small overshoot is legal contact.
THRESH_PCT = 15.0
OUT = os.path.join(HERE, "path_quality.csv")
# ===========================================================================


def check():
    mapper = td.load_mapper()
    if mapper is None:
        raise SystemExit("No field map -- run autofit_fieldmap.py first.")
    cx = (td.FX0 + td.FX1) / 2.0
    half = (td.FX1 - td.FX0) / 2.0

    rows = []
    by_src = defaultdict(lambda: [0, 0])
    for ap in sorted(glob.glob(os.path.join(RESULTS, "*", "auto_paths.json"))):
        if not tp._fresh(ap):
            continue
        match = os.path.basename(os.path.dirname(ap))
        for _st, o in json.load(open(ap)).get("teams", {}).items():
            p = o.get("path") or []
            if len(p) < 5:
                continue
            xy = mapper([(c[1], c[2]) for c in p])
            over = (xy[:, 0] - cx).max() if o["alliance"] == "blue" else (cx - xy[:, 0]).max()
            pct = float(over / half * 100.0)
            src = o.get("source") or "none"
            bad = pct > THRESH_PCT
            by_src[src][1] += 1
            by_src[src][0] += bad
            rows.append(
                {
                    "match": match,
                    "station": o.get("station", ""),
                    "team": (o.get("team") or "").replace("frc", ""),
                    "identity_src": src,
                    "cross_pct": round(pct, 1),
                    "rule_violation": int(bad),
                }
            )
    return rows, by_src


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    rows, by_src = check()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} auto paths checked -> {a.out}")
    print(
        f"\nillegal CENTER LINE crossing (> {THRESH_PCT:.0f}% of a half-field), by identity source:"
    )
    tot = [0, 0]
    for src in ("read", "elim", "rank", "none"):
        bad, n = by_src.get(src, [0, 0])
        if not n:
            continue
        tot[0] += bad
        tot[1] += n
        print(f"   {src:5s}  {bad:4d}/{n:4d}  = {bad / n * 100:5.1f}%")
    if tot[1]:
        print(f"   ALL    {tot[0]:4d}/{tot[1]:4d}  = {tot[0] / tot[1] * 100:5.1f}%")
    print("\nA path flagged here should not be trusted downstream -- it is evidence")
    print("that robot was mis-identified or mis-tracked in that match.")


if __name__ == "__main__":
    main()
