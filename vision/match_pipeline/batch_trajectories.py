"""
Run the YOLO trajectory logger across every downloaded match video.

This is the GPU pipeline: each worker runs the detector on MPS. YOLO inference is
the bottleneck, so a few parallel processes saturate the GPU better than one.
Each match runs in its own process -> a fresh ByteTrack state (no ID bleed).

  python batch_trajectories.py                 # all videos, auto workers
  python batch_trajectories.py --workers 3 --stride 2
  python batch_trajectories.py --only 2026hop_qm1 2026hop_qm2
"""

import argparse
import glob
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")

# EDIT ME (PyCharm one-click). CLI args override.
WORKERS = 5  # GPU jobs in parallel (MPS shared); 3-4 is a good start
STRIDE = 2  # process every Nth frame (60fps footage)
CONF = 0.25
IMGSZ = 960
ONLY = []
FORCE = False


def _weights():
    cands = glob.glob(
        os.path.join(os.path.dirname(HERE), "runs", "detect", "*", "weights", "best.pt")
    )
    if not cands:
        raise SystemExit("No trained weights under runs/detect/*/weights/best.pt")
    return max(cands, key=os.path.getmtime)


def _run_one(args):
    video, outdir, weights, conf, imgsz, stride = args
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    sys.path.insert(0, HERE)
    import trajectories as tj

    t0 = time.time()
    try:
        r = tj.run(video, outdir, weights, conf=conf, imgsz=imgsz, stride=stride, device="mps")
        return (
            os.path.basename(video),
            "ok",
            f"{r['points']} pts, {r['tracks']} tracks ({time.time() - t0:.0f}s)",
        )
    except Exception as e:
        return (
            os.path.basename(video),
            "FAIL",
            f"{e.__class__.__name__}: {e}\n{traceback.format_exc(limit=2)}",
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--conf", type=float, default=CONF)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument("--only", nargs="*", default=ONLY)
    p.add_argument("--force", action="store_true", default=FORCE)
    args = p.parse_args()

    weights = _weights()
    vids = sorted(glob.glob(os.path.join(VIDEOS, "*.mp4")))
    if args.only:
        want = set(args.only)
        vids = [v for v in vids if os.path.splitext(os.path.basename(v))[0] in want]
    jobs = []
    for v in vids:
        mk = os.path.splitext(os.path.basename(v))[0]
        outdir = os.path.join(RESULTS, mk)
        if not args.force and os.path.exists(os.path.join(outdir, "trajectory.csv")):
            continue
        jobs.append((v, outdir, weights, args.conf, args.imgsz, args.stride))

    print(
        f"{len(vids)} videos, {len(jobs)} to run, {args.workers} GPU workers, "
        f"weights={os.path.relpath(weights, HERE)}"
    )
    if not jobs:
        return
    ok = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status, info = fut.result()
            print(f"  [{i}/{len(jobs)}] {name:22s} {status}: {info}")
            ok += status == "ok"
            fail += status != "ok"
    print(f"\nDone: {ok} ok, {fail} failed in {time.time() - t0:.0f}s.")
    print("Next: calibrate once (field_calibrate.py), then batch_auto.py")


if __name__ == "__main__":
    main()
