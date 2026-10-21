"""
Rebuild every match end to end with the current pipeline, resumably.

    python rebuild_all.py                  # every match that has a video and no .rebuilt marker
    python rebuild_all.py --only 2026hop_qm1 2026hop_qm2
    python rebuild_all.py --no-track       # keep existing tracks, redo identity onward
    python rebuild_all.py --no-render      # skip the full-match mp4s (~230 MB each)
    python rebuild_all.py --mark-done 2026hop_qm1 ...   # record matches already rebuilt by hand

Per chunk of matches: ByteTrack (batch_trajectories, parallel) -> then per match identify ->
auto_paths -> render_full_match. A match is finished when results/<match>/.rebuilt exists, so a
rerun after a crash or a stop continues where it left off. A failing match is logged and skipped,
not retried in a loop. Whole-match consumers (steals, fuel_split, teleop_paths, scout_table) are
cheap and read every match, so run them once at the end, not per match.

Identity uses BUMPER_READER from the environment (default easyocr).
"""

import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
VIDEOS = os.path.join(HERE, "videos")
PY = sys.executable


def marker(match):
    return os.path.join(RESULTS, match, ".rebuilt")


def run(cmd, log):
    t0 = time.time()
    r = subprocess.run([PY, *cmd], cwd=HERE, capture_output=True, text=True)
    log(f"    {' '.join(cmd[:3])}: {'ok' if r.returncode == 0 else 'FAIL'} {time.time() - t0:.0f}s")
    if r.returncode:
        log("      " + (r.stderr or r.stdout).strip().splitlines()[-1][:200])
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--chunk", type=int, default=3)
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--mark-done", nargs="*")
    a = ap.parse_args()

    for m in a.mark_done or []:
        open(marker(m), "w").write("hand\n")
    if a.mark_done:
        return

    names = sorted(os.path.splitext(os.path.basename(v))[0] for v in glob.glob(f"{VIDEOS}/*.mp4"))
    todo = [m for m in (a.only or names) if not os.path.exists(marker(m))]
    logf = open(os.path.join(RESULTS, "rebuild_progress.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(f"{time.strftime('%H:%M:%S')} {len(todo)} matches to rebuild (of {len(names)})")
    t_all = time.time()
    for i in range(0, len(todo), a.chunk):
        chunk = todo[i : i + a.chunk]
        log(f"{time.strftime('%H:%M:%S')} chunk {i // a.chunk + 1}: {' '.join(chunk)}")
        if not a.no_track:
            run(["batch_trajectories.py", "--force", "--only", *chunk], log)
        for m in chunk:
            log(f"  {m}")
            ok = run(["identify.py", "--match", m], log)
            ok = ok and run(["auto_paths.py", "--match", m], log)
            if ok and not a.no_render:
                ok = run(["render_full_match.py", "--match", m], log)
            if ok:
                open(marker(m), "w").write(time.strftime("%Y-%m-%d %H:%M\n"))
        done = i + len(chunk)
        el = time.time() - t_all
        log(
            f"{time.strftime('%H:%M:%S')} {done}/{len(todo)} done, "
            f"{el / 60:.0f} min elapsed, ~{el / done * (len(todo) - done) / 60:.0f} min left"
        )


if __name__ == "__main__":
    main()
