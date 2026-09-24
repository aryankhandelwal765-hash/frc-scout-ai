"""
Build the scouting table automatically: one row per (match, team), same schema as a
hand-scouted sheet.

Every field is filled from the CHEAPEST trustworthy source, and every row records
which source was used, because the three are not equally reliable:

  tba     straight from The Blue Alliance. Official. Treat as truth.
  split   a TBA ALLIANCE total apportioned to one robot by its vision activity
          share. The total is exact; the split is an estimate.
  vision  measured from the broadcast only. Carries the identity and detection
          error described in the project notes.
  --      not derivable from either. Subjective judgement (driver skill, notes,
          reliability) stays a human's job; this file will never invent it.

Deliberately NOT guessed: any column whose value depends on field zones that have
not been defined yet in canonical field coordinates. Those emit blank with source
"needs-zones" rather than a plausible-looking number.

PyCharm: click Run.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)

RESULTS = os.path.join(HERE, "results")
TRUTH = os.path.join(HERE, "truth")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
EVENT = "2026hop"
OUT = os.path.join(HERE, "scout_table.csv")
# ===========================================================================

# sheet column -> where it comes from. Kept explicit so the gaps stay visible.
SOURCES = {
    "match":         "tba",
    "team":          "tba",
    "alliance":      "tba",
    "station":       "tba",      # TBA lists team keys in driver-station order
    "auto_climb":    "tba",      # autoTowerRobot{1,2,3}
    "climb":         "tba",      # endGameTowerRobot{1,2,3}
    "failed_climb":  "--",       # TBA records what was achieved, not what was tried
    "alliance_auto_fuel":    "tba",    # hubScore.autoCount
    "alliance_teleop_fuel":  "tba",    # hubScore.teleopCount
    "alliance_endgame_fuel": "tba",    # hubScore.endgameCount
    "alliance_total_fuel":   "tba",    # hubScore.totalCount
    "auto_fuel":     "split",    # alliance autoCount x this robot's activity share
    "teleop_fuel":   "split",
    "endgame_fuel":  "split",
    "total_fuel":    "split",
    "share_covered": "split",    # how much alliance weight was attributable
    "auto_start":    "vision",   # start position -> station
    "identity_src":  "vision",   # read / elim / rank -- how the team was identified
    "baseline":      "--",       # no scoring meaning in REBUILT: Table 6-4 has no
                                 # entry for leaving the ROBOT STARTING LINE. The
                                 # column is vestigial from an earlier game.
    "auto_bump":     "vision",   # auto path entered a BUMP zone
    "auto_trench":   "vision",   # auto path entered a TRENCH zone
    "tele_bump":     "vision",   # teleop path entered a BUMP zone
    "tele_trench":   "vision",   # teleop path entered a TRENCH zone
    "tele_bump_visits":   "vision",   # separate entries, not just yes/no
    "tele_trench_visits": "vision",
    "tele_hub_visits":    "vision",
    "tele_top_zones":     "vision",   # where teleop was actually spent
    "path_sequence": "vision",   # ordered zones the auto path visited
    "defense":       "--",
    "tele_passing":  "--",
    "hp_accuracy":   "--",
    "reliability":   "--",
    "driver_skill":  "--",
    "notes":         "--",
}

COLUMNS = list(SOURCES.keys())


def tower_level(v):
    """TBA's tower field -> level string, or None for 'did not climb'.

    Careful: TBA serialises the empty case as the STRING "None", not a JSON null.
    bool("None") is True, so a plain truthiness test scores every robot as a
    climber -- it reported 840/840 here against a real rate of 13/840.
    """
    if v is None:
        return None
    s = str(v).strip()
    return None if s in ("", "None", "null", "Unknown") else s


def _zone_facts(o, mapper, zones, td):
    """bump/trench usage and the zone sequence for one robot's auto path."""
    p = o.get("path") or []
    if len(p) < 5 or mapper is None:
        return {}
    fi = zones.to_field_inches(mapper([(c[1], c[2]) for c in p]))
    return {"auto_bump": int(zones.used(fi, "Bump")),
            "auto_trench": int(zones.used(fi, "Trench")),
            "path_sequence": " -> ".join(zones.zone_sequence(fi))}


def _auto_paths(match):
    p = os.path.join(RESULTS, match, "auto_paths.json")
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p)).get("teams", {})
    except Exception:
        return {}


def build(event=EVENT):
    try:
        import topdown as td
        import zones
        mapper = td.load_mapper()
    except Exception as e:
        print(f"  (zone columns unavailable: {e.__class__.__name__}: {e})")
        mapper, zones, td = None, None, None
    tele = {}
    tp_path = os.path.join(HERE, "teleop_facts.json")
    if os.path.exists(tp_path):
        try:
            tele = json.load(open(tp_path))
        except Exception as e:
            print(f"  (teleop facts unreadable: {e})")
    else:
        print("  (no teleop_facts.json -- run teleop_paths.py to fill tele_* columns)")
    fuel = {}
    fp = os.path.join(HERE, "fuel_split.csv")
    if os.path.exists(fp):
        with open(fp) as f:
            for r in csv.DictReader(f):
                fuel[(r["match"], r["team"])] = r
    else:
        print("  (no fuel_split.csv -- run fuel_split.py to fill the fuel columns)")
    rows = []
    counts = defaultdict(int)
    for tp in sorted(glob.glob(os.path.join(TRUTH, f"{event}_*.json"))):
        truth = json.load(open(tp))
        match = truth.get("key") or os.path.splitext(os.path.basename(tp))[0]
        sb = truth.get("score_breakdown") or {}
        ap = _auto_paths(match)
        # station -> vision record, so a team can be matched to what vision saw
        by_station = {o.get("station"): o for o in ap.values() if o.get("station")}

        for al in ("blue", "red"):
            teams = truth.get(f"{al}_teams") or []
            hub = ((sb.get(al) or {}).get("hubScore") or {})
            d = sb.get(al) or {}
            for i, team in enumerate(teams, start=1):
                v = by_station.get(f"{al}{i}") or {}
                r = {c: "" for c in COLUMNS}
                r.update({
                    "match": match,
                    "team": str(team).replace("frc", ""),
                    "alliance": al,
                    "station": i,
                    # TBA records the tower level achieved, or None. None = no climb.
                    "auto_climb": int(tower_level(d.get(f"autoTowerRobot{i}")) is not None),
                    "climb": int(tower_level(d.get(f"endGameTowerRobot{i}")) is not None),
                    "alliance_auto_fuel": hub.get("autoCount", ""),
                    "alliance_teleop_fuel": hub.get("teleopCount", ""),
                    "alliance_endgame_fuel": hub.get("endgameCount", ""),
                    "alliance_total_fuel": hub.get("totalCount", ""),
                    "identity_src": v.get("source", ""),
                    "auto_start": v.get("station", ""),
                })
                if v:
                    r.update(_zone_facts(v, mapper, zones, td))
                fu = fuel.get((match, str(team).replace("frc", "")))
                if fu:
                    r.update({k: fu[k] for k in
                              ("auto_fuel", "teleop_fuel", "endgame_fuel",
                               "total_fuel", "share_covered") if k in fu})
                    counts["with_fuel"] += 1
                tf = (tele.get(match) or {}).get(str(team).replace("frc", ""))
                if tf:
                    r.update({k: tf[k] for k in tf if k in COLUMNS})
                    counts["with_teleop"] += 1
                counts["rows"] += 1
                counts["with_vision"] += bool(v)
                counts[f"id_{v.get('source', 'none')}"] += 1
                rows.append(r)
    return rows, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--event", default=EVENT)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    rows, counts = build(a.event)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} team-match rows -> {a.out}")
    print(f"  rows with vision data: {counts['with_vision']} "
          f"({100.0 * counts['with_vision'] / max(len(rows), 1):.0f}%)")
    print(f"  rows with fuel estimates: {counts['with_fuel']} "
          f"({100.0 * counts['with_fuel'] / max(len(rows), 1):.0f}%)")
    print(f"  rows with teleop paths: {counts['with_teleop']} "
          f"({100.0 * counts['with_teleop'] / max(len(rows), 1):.0f}%)")
    print("  identity source:", {k[3:]: v for k, v in counts.items()
                                 if k.startswith("id_")})
    bysrc = defaultdict(list)
    for c, s in SOURCES.items():
        bysrc[s].append(c)
    print("\ncolumn sources:")
    for s in ("tba", "split", "vision", "needs-zones", "--"):
        print(f"  {s:12s} {', '.join(bysrc[s])}")


if __name__ == "__main__":
    main()
