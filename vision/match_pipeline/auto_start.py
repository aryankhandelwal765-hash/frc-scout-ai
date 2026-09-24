"""
Find the exact frame auto begins, per match, by reading the match timer.

Why this is needed. preprocess.match_window() samples the timer only every
SAMPLE_EVERY (2.0) seconds, so the trim point it finds is accurate to +-2s. Auto's
position inside the trimmed video is therefore PAD +- 2s, not a constant -- yet
auto_paths.py assumed a flat AUTO_START_SEC = 2.0 for every match. Measured across
110 matches, 61% were off by more than a second. That smears every start position
and is why so many auto paths appear to begin mid-period.

How it works. During auto the timer counts DOWN from AUTO_SECONDS. Read it at a
handful of frames and each reading independently implies a start frame:

    start = frame - (AUTO_LEN - timer) * fps

Take the median of those estimates, so one bad OCR cannot move the answer. Reading
the countdown is far more robust than hunting for a single transition frame.

The timer box: TIMER_BOX_FRAC in preprocess is a fraction of the FULL broadcast
frame, but these videos are the top view, which is only VIEW_RECTS["top"] height of
it. The y fractions are rescaled accordingly -- using them raw reads nothing.

PyCharm: click Run to scan every match and write auto_start.json.
"""

import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
import preprocess as pp

VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "auto_start.json")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
ONLY = []            # match keys; [] = every video
AUTO_LEN = 20        # auto length in seconds, READ off the broadcast timer (it
                     # counts 20 -> 1 then jumps to 2:20 teleop). Not 15.
SCAN_SEC = 26.0      # how far into the video to look for the countdown
STRIDE = 20          # frames between timer OCR samples
WORKERS = 6
# ===========================================================================


def timer_box(shape):
    """TIMER_BOX_FRAC rescaled from the full frame into the top view."""
    h, w = shape[:2]
    fx1, fy1, fx2, fy2 = pp.TIMER_BOX_FRAC
    th = pp.VIEW_RECTS["top"][3]
    return (int(fx1 * w), int(fy1 / th * h), int(fx2 * w), int(fy2 / th * h))


def detect(match, auto_len=AUTO_LEN, scan_sec=SCAN_SEC, stride=STRIDE):
    """-> dict(match, start_frame, fps, n_reads, spread) or None."""
    vid = os.path.join(VIDEOS, f"{match}.mp4")
    if not os.path.exists(vid):
        return None
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    ok, f0 = cap.read()
    if not ok:
        cap.release()
        return None
    box = timer_box(f0.shape)

    ests, reads = [], []
    f = 0
    limit = int(scan_sec * fps)
    while f < limit:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        t = pp.read_timer(frame, box)
        # only the auto countdown: 1..auto_len. Teleop reads >= 60 and is ignored.
        if t is not None and 1 <= t <= auto_len:
            reads.append((f, t))
            ests.append(f - (auto_len - t) * fps)
        f += stride
    cap.release()
    if len(ests) < 4:
        return None
    ests = np.array(ests, float)
    return {"match": match, "start_frame": int(round(float(np.median(ests)))),
            "fps": float(fps), "n_reads": len(ests),
            "spread": float(np.percentile(ests, 90) - np.percentile(ests, 10))}


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    matches = ONLY or sorted(os.path.splitext(os.path.basename(p))[0]
                             for p in glob.glob(os.path.join(VIDEOS, "*.mp4")))
    print(f"{len(matches)} matches, auto_len={AUTO_LEN}s")
    out = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(detect, m): m for m in matches}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:
                print(f"  {futs[fut]}: {e.__class__.__name__}: {e}")
                continue
            if r:
                out[r["match"]] = r
            if i % 20 == 0:
                print(f"  ... {i}/{len(matches)}")
    json.dump(out, open(OUT, "w"), indent=2)
    if out:
        sf = np.array([v["start_frame"] for v in out.values()])
        sp = np.array([v["spread"] for v in out.values()])
        print(f"\ndetected on {len(out)}/{len(matches)} matches -> {OUT}")
        print(f"  start_frame: min={sf.min()} median={np.median(sf):.0f} "
              f"max={sf.max()} std={sf.std():.0f}")
        print(f"  per-match estimate spread (p90-p10, frames): "
              f"median={np.median(sp):.0f} max={sp.max():.0f}")
        print("  A small spread means the countdown reads agree -- that is the "
              "detector checking itself.")
    else:
        print("no matches detected -- check the timer box / AUTO_LEN")


if __name__ == "__main__":
    main()
