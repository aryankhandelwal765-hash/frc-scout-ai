"""
REPL labeller: shows real zone sequences from match data and prompts you to label each one.
Labels become few-shot examples baked into interpret_auto.py's system prompt.

Usage:
  python label_auto.py              # label all unlabelled sequences
  python label_auto.py --count 5   # preview 5 without saving (dry run)
  python label_auto.py --all       # re-label already-labelled sequences
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import topdown as td
import zones
from topdown_paths import _fresh, to_canonical_fi

RESULTS = os.path.join(HERE, "results")
TRAINING_DIR = os.path.join(HERE, "training")
OUTPUT = os.path.join(TRAINING_DIR, "auto_sequences.jsonl")


def _start_delay_s(pts_with_t, threshold_in=30.0):
    """Seconds from first detection to first move > threshold_in from start position."""
    if len(pts_with_t) < 2:
        return 0.0
    t0, x0, y0 = pts_with_t[0]
    for t, x, y in pts_with_t[1:]:
        if ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 > threshold_in:
            return round(t - t0, 2)
    return round(pts_with_t[-1][0] - t0, 2)


def load_sequences(mapper):
    """Return list of dicts, one per robot-in-match, with zone_seq and timing."""
    seqs = []
    for ap_path in sorted(glob.glob(os.path.join(RESULTS, "*", "auto_paths.json"))):
        if not _fresh(ap_path):
            continue
        match = os.path.basename(os.path.dirname(ap_path))
        smooth_path = os.path.join(os.path.dirname(ap_path), "smoothed_auto_paths.json")
        use_path = smooth_path if os.path.exists(smooth_path) else ap_path

        try:
            data = json.load(open(use_path))
        except Exception:
            continue

        for station, o in data.get("teams", {}).items():
            team = o.get("team")
            path = o.get("path", [])
            if not team or len(path) < 5:
                continue

            if o.get("coords") == "field_in":
                fi = np.array([[fx, fy] for _t, fx, fy in path], float)
                pts = [(row[0], row[1], row[2]) for row in path]
            else:
                canvas = mapper([(c[1], c[2]) for c in path])
                fi = zones.to_field_inches(np.asarray(canvas))
                pts = [(path[i][0], fi[i, 0], fi[i, 1]) for i in range(len(fi))]

            # Rotate into canonical blue frame before extracting zone sequence
            fi_can = to_canonical_fi(fi, o.get("alliance", "blue"))
            zone_seq = zones.zone_sequence(fi_can.tolist(), min_dwell=3)
            delay = _start_delay_s(pts)
            duration = round(path[-1][0] - path[0][0], 2) if len(path) >= 2 else 0.0

            seqs.append({
                "match": match,
                "station": station,
                "team": team,
                "alliance": o.get("alliance", ""),
                "source": o.get("source", "none"),
                "zone_seq": zone_seq,
                "start_delay_s": delay,
                "duration_s": duration,
            })
    return seqs


def _already_labelled():
    done = set()
    if os.path.exists(OUTPUT):
        for line in open(OUTPUT):
            try:
                r = json.loads(line)
                done.add((r["match"], r["station"]))
            except Exception:
                pass
    return done


def _show(seq, idx, total):
    team_str = seq["team"].replace("frc", "#") if seq["team"] else "unknown"
    print(f"\n{'='*62}")
    print(f"[{idx}/{total}]  {seq['match']} / {seq['station']} / {team_str}"
          f"  ({seq['source']})")
    print(f"Alliance: {seq['alliance']}")
    zone_str = " → ".join(seq["zone_seq"]) if seq["zone_seq"] else "(no zones detected)"
    print(f"Zones:    {zone_str}")
    print(f"Start delay: {seq['start_delay_s']}s   Duration: {seq['duration_s']}s")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=0,
                   help="dry run: preview this many sequences without saving")
    p.add_argument("--all", action="store_true",
                   help="re-label already-labelled sequences too")
    a = p.parse_args()

    mapper = td.load_mapper()
    if mapper is None:
        raise SystemExit("No field map. Run multi_video_apriltag_calib.py first.")

    print("Loading zone sequences from results/ ...")
    seqs = load_sequences(mapper)
    if not seqs:
        raise SystemExit("No valid auto_paths.json found in results/")

    done = set() if a.all else _already_labelled()
    pending = [s for s in seqs
               if (s["match"], s["station"]) not in done and s["zone_seq"]]

    if a.count:
        show_n = min(a.count, len(pending))
        for i, seq in enumerate(pending[:show_n]):
            _show(seq, i + 1, show_n)
        print(f"({len(pending)} unlabelled sequences total, "
              f"{len(seqs) - len(pending)} already done)")
        return

    if not pending:
        print(f"All {len(seqs)} sequences already labelled. Use --all to re-label.")
        return

    os.makedirs(TRAINING_DIR, exist_ok=True)
    saved = 0
    print(f"{len(pending)} sequences to label.")
    print("  [Enter] skip field to abort entry | [s] skip this robot | [q] quit\n")

    try:
        with open(OUTPUT, "a") as f:
            for i, seq in enumerate(pending):
                _show(seq, i + 1, len(pending))

                name = input("  Routine name [e.g. Left 3-Piece]: ").strip()
                if name.lower() == "q":
                    break
                if name.lower() == "s" or not name:
                    print("  skipped")
                    continue

                desc = input("  One-sentence description: ").strip()
                if not desc:
                    print("  skipped (no description)")
                    continue

                side_raw = input("  Start side [L/C/R]: ").strip().upper()
                side_map = {"L": "left", "C": "centre", "R": "right"}
                if side_raw not in side_map:
                    print("  skipped (invalid side)")
                    continue
                start_side = side_map[side_raw]

                delayed_raw = input("  Delayed start? [y/N]: ").strip().lower()
                delayed = delayed_raw == "y"

                record = {
                    "match": seq["match"],
                    "station": seq["station"],
                    "team": seq["team"],
                    "alliance": seq["alliance"],
                    "zone_seq": seq["zone_seq"],
                    "start_delay_s": seq["start_delay_s"],
                    "duration_s": seq["duration_s"],
                    "label": {
                        "name": name,
                        "description": desc,
                        "start_side": start_side,
                        "delayed": delayed,
                    },
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                saved += 1
                print(f"  --> saved ({saved})")

    except (KeyboardInterrupt, EOFError):
        pass

    print(f"\nDone. {saved} examples saved to {OUTPUT}")
    print("These will be used as few-shot examples in interpret_auto.py")


if __name__ == "__main__":
    main()
