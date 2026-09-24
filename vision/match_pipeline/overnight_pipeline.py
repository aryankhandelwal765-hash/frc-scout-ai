"""
Overnight trajectory correction pipeline.

Run the night before competition. All phases are independent and resumable:
if a phase already produced its output it is skipped unless --force is given.

Phase 1  multi_video_apriltag_calib.py
         Collect AprilTag observations from every match video, solve the
         best-fit camera pose across the event, replace the fieldmap in
         calib.json with a physically-grounded homography.
         Biggest accuracy lever -- gets the perspective geometry right.

Phase 2  fit_2d_correction.py
         Fit a 2D polynomial (raw_X, raw_Y) -> (dX, dY) from ~300 robot
         starting-line observations. Removes residual systematic error
         that varies across both field axes.

Phase 3  smooth_trajectories.py
         Apply Phase 1+2 correction + Kalman [x,y,vx,vy] filter to every
         trajectory.csv.  Gap-fills occlusion windows (ball possession,
         robots crossing). Writes smoothed_trajectory.csv + smoothed_auto_paths.json.

Phase 4  Validate + re-render
         Re-render topdown_auto.png for every match (uses smoothed_auto_paths.json
         automatically). Run validate_paths.py and print the accuracy report.

Phase 5  Auto book (LLM-assisted written scouting report)
         5a: interpret_auto.py -- LLM reads each robot's zone sequence and writes
             a structured interpretation (name, description, start_side, delayed).
         5b: team_routes.py --json-only -- spatial + delay clustering -> team_routes.json.
         5c: auto_book.py -- synthesises per-team written auto entries -> auto_book/*.md.
         Requires Ollama running. Run label_auto.py first for better few-shot examples.

Usage:
  python overnight_pipeline.py              # run all phases
  python overnight_pipeline.py --skip 1 2  # skip phases 1 and 2
  python overnight_pipeline.py --only 3 4  # only run phases 3 and 4
  python overnight_pipeline.py --only 5    # only written auto book
  python overnight_pipeline.py --force     # redo everything even if outputs exist
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(HERE, "calib.json")
RESULTS = os.path.join(HERE, "results")

PYTHON = sys.executable


def _elapsed(t0):
    s = int(time.time() - t0)
    return f"{s // 60}m {s % 60}s"


def _run(script, *args, label=""):
    cmd = [PYTHON, os.path.join(HERE, script)] + list(args)
    label = label or script
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=HERE)
    elapsed = _elapsed(t0)
    if result.returncode != 0:
        print(f"\n[FAILED] {label} exited {result.returncode} after {elapsed}")
        return False
    print(f"\n[OK] {label} finished in {elapsed}")
    return True


def phase1_done():
    if not os.path.exists(CALIB):
        return False
    calib = json.load(open(CALIB))
    gh = calib.get("ground_homography", {})
    return bool(gh and "H" in gh and gh.get("n_corners", 0) > 20)


def phase2_done():
    if not os.path.exists(CALIB):
        return False
    return bool(json.load(open(CALIB)).get("correction_2d"))


def phase3_done():
    import glob
    smoothed = glob.glob(os.path.join(RESULTS, "*", "smoothed_trajectory.csv"))
    raw = glob.glob(os.path.join(RESULTS, "*", "trajectory.csv"))
    return len(smoothed) >= len(raw) * 0.9  # 90% smoothed = "done"


def phase4_render(force=False):
    """Re-render topdown_auto.png for every match that has smoothed_auto_paths.json."""
    sys.path.insert(0, HERE)
    import glob
    import topdown as td
    import topdown_paths as tp

    mapper = td.load_mapper()
    if mapper is None:
        print("  No fieldmap, cannot render")
        return

    paths = sorted(glob.glob(os.path.join(RESULTS, "*", "smoothed_auto_paths.json")))
    ok = 0
    for p in paths:
        match = os.path.basename(os.path.dirname(p))
        out = os.path.join(RESULTS, match, "topdown_auto.png")
        if not force and os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(p):
            continue
        result = tp.render_match(match, mapper)
        if result:
            ok += 1
    print(f"  Rendered {ok} topdown_auto.png files")
    # Per-team aggregate
    tp.render_team_aggregate(mapper)


def phase4_validate():
    """Run validate_paths.py and print the center-line violation report."""
    script = os.path.join(HERE, "validate_paths.py")
    if not os.path.exists(script):
        print("  validate_paths.py not found, skipping")
        return
    subprocess.run([PYTHON, script], cwd=HERE)


def phase5_done():
    all_md = os.path.join(HERE, "auto_book", "_all.md")
    if not os.path.exists(all_md):
        return False
    # Done if _all.md is newer than the newest auto_interp.json
    import glob
    interps = glob.glob(os.path.join(RESULTS, "*", "auto_interp.json"))
    if not interps:
        return False
    newest_interp = max(os.path.getmtime(f) for f in interps)
    return os.path.getmtime(all_md) > newest_interp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip", nargs="*", type=int, default=[],
                   help="phase numbers to skip, e.g. --skip 1 2")
    p.add_argument("--only", nargs="*", type=int, default=[],
                   help="run ONLY these phase numbers")
    p.add_argument("--force", action="store_true",
                   help="redo all phases even if outputs already exist")
    a = p.parse_args()

    def should_run(n):
        if a.only:
            return n in a.only
        return n not in a.skip

    wall_t0 = time.time()
    print("\nOVERNIGHT PIPELINE")
    print(f"  calib: {CALIB}")
    print(f"  results: {RESULTS}")

    # --- Phase 1 ---
    if should_run(1):
        if not a.force and phase1_done():
            print("\n[Phase 1] Already done (ground_homography exists). --force to redo.")
        else:
            ok = _run("multi_video_apriltag_calib.py",
                      label="Phase 1: Multi-video AprilTag camera pose")
            if not ok:
                print("Phase 1 failed -- phases 2-4 may produce worse results")
    else:
        print("\n[Phase 1] Skipped")

    # --- Phase 2 ---
    if should_run(2):
        if not a.force and phase2_done():
            print("\n[Phase 2] Already done (correction_2d exists). --force to redo.")
        else:
            ok = _run("fit_2d_correction.py",
                      label="Phase 2: 2D residual correction polynomial")
            if not ok:
                print("Phase 2 failed -- continuing without 2D correction")
    else:
        print("\n[Phase 2] Skipped")

    # --- Phase 3 ---
    if should_run(3):
        force_arg = ["--force"] if a.force else []
        if not a.force and phase3_done():
            print("\n[Phase 3] Already done (smoothed_trajectory.csv exists for all matches). "
                  "--force to redo.")
        else:
            ok = _run("smooth_trajectories.py", *force_arg,
                      label="Phase 3: Kalman smoothing + gap fill")
            if not ok:
                print("Phase 3 failed -- topdown renders will use raw paths")
    else:
        print("\n[Phase 3] Skipped")

    # --- Phase 4 ---
    if should_run(4):
        print(f"\n{'='*60}")
        print("  Phase 4: Re-render topdown paths + validate")
        print(f"{'='*60}")
        t0 = time.time()
        phase4_render(force=a.force)
        phase4_validate()
        print(f"\n[OK] Phase 4 finished in {_elapsed(t0)}")
    else:
        print("\n[Phase 4] Skipped")

    # --- Phase 5 ---
    if should_run(5):
        if not a.force and phase5_done():
            print("\n[Phase 5] Already done (auto_book/_all.md is current). --force to redo.")
        else:
            print(f"\n{'='*60}")
            print("  Phase 5: LLM auto interpretation + written scouting book")
            print(f"{'='*60}")
            t0 = time.time()
            force_arg = ["--force"] if a.force else []
            ok5a = _run("interpret_auto.py", *force_arg,
                        label="Phase 5a: LLM zone-sequence interpretation per match")
            if not ok5a:
                print("Phase 5a failed -- auto_book will use heuristic fallback")
            ok5b = _run("team_routes.py", "--json-only",
                        label="Phase 5b: Cluster routes + export team_routes.json")
            if not ok5b:
                print("Phase 5b failed -- auto_book cannot run without team_routes.json")
            else:
                ok5c = _run("auto_book.py", *force_arg,
                            label="Phase 5c: Written auto scouting book")
                if not ok5c:
                    print("Phase 5c failed")
                _run("render_zone_paths.py", *force_arg,
                     label="Phase 5d: Zone sequence PNGs per robot")
            print(f"\n[OK] Phase 5 finished in {_elapsed(t0)}")
    else:
        print("\n[Phase 5] Skipped")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {_elapsed(wall_t0)}")
    print(f"{'='*60}")
    print("\nResults in:")
    print(f"  {RESULTS}/*/smoothed_auto_paths.json  -- smoothed paths (field inches)")
    print(f"  {RESULTS}/*/topdown_auto.png          -- re-rendered match renders")
    print(f"  {os.path.join(HERE, 'topdown_paths/')}  -- per-team aggregate pages")
    print(f"  {os.path.join(HERE, 'auto_book/_all.md')}  -- written auto scouting book")
    print(f"\nIf validate_paths.py shows worse center-line violations than before,")
    print("run: python overnight_pipeline.py --only 1 --force")
    print("to re-solve Phase 1 with different parameters, then re-run 2-4.")


if __name__ == "__main__":
    main()
