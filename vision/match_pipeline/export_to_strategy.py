"""
Export vision's measured scouting data into strategy, the strategy system.

strategy (the strategy app at the repo root) already has the consumer: its
scout/training/build_scouting_features.py reads a per-team-per-match CSV and
aggregates it into team_scouting_features.csv, which feeds the quant model and the
qwen3.6 strategy-captain reads. That input CSV has always been a 0-byte placeholder
because it was meant to be filled by hand.

This writes it from video instead.

The schemas already agree -- strategy expects exactly the hand-scouting columns
(team, auto_fuel, climb, failed_climb, defense, tele_bump, tele_trench,
tele_passing, hp_accuracy, reliability, driver_skill) and scout_table.py emits that
same shape.

IMPORTANT about gaps. strategy's aggregator reads missing columns as an empty Series
and its iqm() helper returns 0 for empty. So an unmeasured column does not error --
it silently becomes 0.0, which reads as "this team never does that" rather than
"we did not measure it". That is a real way to mislead a pick list, so this script
refuses to emit a column it cannot measure, and prints exactly which ones strategy
will therefore zero-fill.

PyCharm: click Run.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
# Resolved relative to this repo (match_pipeline/ -> vision/ -> repo root),
# same as interpret_auto.py and auto_book.py. Override with --strategy.
STRATEGY_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "strategy")
TABLE = os.path.join(HERE, "scout_table.csv")
OUT_NAME = "vision_2026hop.csv"   # lands in strategy/scout/data/scouting/
WRITE = True                          # False = report only, touch nothing
# ===========================================================================

# MEASURED RELIABILITY GATE.
#
# strategy aggregates whatever it is given into a per-team feature with iqm(), which
# produces a stable-looking number from ANY input -- including pure noise. So the
# gate has to be here, not there.
#
# Split-half reliability (each team's matches split odd/even, the halves
# correlated, Spearman-Brown corrected) says whether a metric describes the TEAM or
# just that match. Measured on 2026hop:
#
#   climb                0.92   TBA data
#   steal_success        0.68
#   total_fuel           0.67
#   tele_bump_visits     0.61
#   steals               0.60
#   teleop_fuel          0.59
#   tele_trench_visits   0.29   NOISE
#   auto_fuel            0.22   NOISE
#   tele_bump (binary)  -0.35   NOISE
#   tele_trench (binary) 0.30   NOISE
#   auto_bump (binary)  -0.38   NOISE
#   auto_trench (binary) 0.29   NOISE
#
# The binary bump/trench flags are the worst of the lot: ~80% of rows are 1, so
# they saturate and carry no team signal at all. Their COUNT versions survive,
# which is the tell -- the information is in how often, not whether.
#
# Cross-checked against 73 hand-scouted teams: the binary flags correlate r=+0.07
# to -0.13 with human scouting. Population means match (tele_bump 0.76 vs 0.78) but
# team ordering is random, exactly as the reliability numbers predict.
MIN_RELIABILITY = 0.55

RELIABILITY = {
    "climb": 0.92, "auto_climb": 0.92, "steal_success": 0.68, "total_fuel": 0.67,
    "tele_bump_visits": 0.61, "steals": 0.60, "teleop_fuel": 0.59,
    "tele_trench_visits": 0.29, "share_covered": 0.27, "auto_fuel": 0.22,
    "tele_bump": -0.35, "tele_trench": 0.30, "auto_bump": -0.38, "auto_trench": 0.29,
}

# strategy column -> how vision fills it. None = we cannot measure it honestly.
MAPPING = {
    "team":          ("team", "tba"),
    "climb":         ("climb", "tba"),
    "auto_climb":    ("auto_climb", "tba"),
    "auto_fuel":     ("auto_fuel", "split"),
    "tele_bump":     ("tele_bump", "vision"),
    "tele_trench":   ("tele_trench", "vision"),
    "failed_climb":  (None, "not-measurable"),
    "defense":       (None, "not-measurable"),
    "tele_passing":  (None, "not-measurable"),
    "hp_accuracy":   (None, "not-measurable"),
    "reliability":   (None, "not-measurable"),
    "driver_skill":  (None, "not-measurable"),
    # extras strategy does not ask for but its trait engine can use
    "teleop_fuel":        ("teleop_fuel", "split"),
    "total_fuel":         ("total_fuel", "split"),
    "share_covered":      ("share_covered", "split"),
    "auto_bump":          ("auto_bump", "vision"),
    "auto_trench":        ("auto_trench", "vision"),
    "tele_bump_visits":   ("tele_bump_visits", "vision"),
    "tele_trench_visits": ("tele_trench_visits", "vision"),
}


def load(path=TABLE):
    if not os.path.exists(path):
        raise SystemExit(f"No {path} -- run scout_table.py first.")
    with open(path) as f:
        return list(csv.DictReader(f))


def build(rows):
    """Only columns that are measured AND reliable enough to describe a team."""
    cols = [c for c, (src, _) in MAPPING.items()
            if src and RELIABILITY.get(c, 1.0) >= MIN_RELIABILITY]
    out = []
    for r in rows:
        # a row identified only by the unverified station-order guess may describe
        # the wrong robot, so mark it rather than letting it pass as measured
        rec = {c: r.get(MAPPING[c][0], "") for c in cols}
        rec["identity_src"] = r.get("identity_src", "")
        out.append(rec)
    return cols + ["identity_src"], out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default=STRATEGY_DIR)
    p.add_argument("--out", default=OUT_NAME)
    p.add_argument("--dry-run", action="store_true", default=not WRITE)
    a = p.parse_args()

    rows = load()
    cols, recs = build(rows)

    dest_dir = os.path.join(a.strategy, "scout", "data", "scouting")
    dest = os.path.join(dest_dir, a.out)

    print(f"{len(recs)} team-match rows from {len(set(r['team'] for r in recs))} teams")
    print(f"\nEXPORTED ({len(cols)} columns): {', '.join(cols)}")

    missing = {c: why for c, (src, why) in MAPPING.items() if not src}
    unreliable = {c: RELIABILITY[c] for c, (src, _) in MAPPING.items()
                  if src and RELIABILITY.get(c, 1.0) < MIN_RELIABILITY}
    print(f"\nWITHHELD as unreliable ({len(unreliable)}) -- split-half reliability "
          f"below {MIN_RELIABILITY}:")
    for c, v in sorted(unreliable.items(), key=lambda kv: kv[1]):
        print(f"   {c:20s} reliability {v:+.2f}  (measures the match, not the team)")
    print(f"\nNOT MEASURABLE ({len(missing)}):")
    for c, why in sorted(missing.items(), key=lambda kv: kv[1]):
        print(f"   {c:20s} {why}")
    print("\n  strategy's build_scouting_features.py will read each missing column as")
    print("  an empty Series and iqm() will return 0.0 -- i.e. it will look like the")
    print("  team NEVER did that thing, not like it was unmeasured. Do not treat")
    print("  those aggregates as evidence.")

    if a.dry_run:
        print(f"\n[dry run] would write {dest}")
        return
    if not os.path.isdir(dest_dir):
        raise SystemExit(f"strategy scouting dir not found: {dest_dir}")
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)
    print(f"\nwrote {dest}")
    print("\nNext, in strategy:")
    print(f"  1. point build_scouting_features.py INPUT_FILE at {a.out}")
    print("  2. run it -> team_scouting_features.csv")
    print("  3. that feeds the quant model + the strategy-captain read")


if __name__ == "__main__":
    main()
