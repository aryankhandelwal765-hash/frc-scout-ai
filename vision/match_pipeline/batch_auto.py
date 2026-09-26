"""
Run auto_paths.py over every match that has a trajectory, then build the
preferred (aggregated) auto paths per team.

  python batch_auto.py                      # all matches with trajectory.csv
  python batch_auto.py --auto-seconds 15

Needs: trajectory.csv per match (batch_trajectories.py) + calib.json
(field_calibrate.py). CPU-only + fast, so it runs serially.
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, "results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--auto-seconds", type=float, default=15.0)
    p.add_argument("--skip-preferred", action="store_true")
    args = p.parse_args()

    import auto_paths as ap

    matches = sorted(
        os.path.basename(os.path.dirname(x))
        for x in glob.glob(os.path.join(RESULTS, "*", "trajectory.csv"))
    )
    print(f"{len(matches)} matches with trajectories")
    ok = fail = 0
    for m in matches:
        try:
            ap.run(m, auto_seconds=args.auto_seconds)
            ok += 1
        except SystemExit as e:
            print(f"  {m}: skip ({e})")
            fail += 1
        except Exception as e:
            print(f"  {m}: FAIL {e.__class__.__name__}: {e}")
            fail += 1
    print(f"\nauto_paths: {ok} ok, {fail} skipped/failed")
    if not args.skip_preferred:
        import preferred_paths as pp

        pp.run()


if __name__ == "__main__":
    main()
