"""
Run the vision scoring estimator across every downloaded match video, in parallel.

Uses scoring_estimator.run() (pure CV + scoreboard OCR -> per-alliance ball totals
+ per-robot activity split). No trained weights, no per-video goal zones, so it
batches cleanly. Each video -> results/<match_key>/estimate.json.

Tuned for a fat machine: many videos run concurrently (one process each). The
estimator is single-threaded CV+tesseract, so N processes ~= N cores of throughput.

Usage:
  python batch_run.py                       # all videos, auto worker count
  python batch_run.py --workers 12
  python batch_run.py --video --workers 6   # also write annotated.mp4 (slower)
  python batch_run.py --only 2026hop_qm1 2026hop_qm2
"""

import argparse
import glob
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
# scoring_estimator.py lives next to this script (on sys.path as the script dir).
sys.path.insert(0, PARENT)  # import the estimator from the project root
VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")

# ===========================================================================
# EDIT ME (PyCharm: click Run — no terminal args needed). CLI args override.
# ===========================================================================
WORKERS = 16  # parallel processes (18-core M5 Pro: go wide); 0 = auto
WRITE_VIDEO = False  # True = also render annotated.mp4 per match (slower)
OCR_EVERY = 10  # OCR the scoreboard every N frames
SLOTS = 3  # robots per alliance
STRIDE = 2  # process every Nth frame (footage is 60fps; 2 = ~2x faster)
FORCE = False  # True = re-run matches that already have estimate.json
ONLY = []  # e.g. ["2026hop_qm1", "2026hop_qm2"]; [] = all videos
# ===========================================================================


def _run_one(args):
    """Worker: run the estimator on one video. Runs in its own process."""
    video_path, outdir, ocr_every, k_slots, write_video, crop, field_mid, stride = args
    # pin each worker to a single math/CV thread: with many processes, OpenCV's
    # internal thread pool oversubscribes the cores and thrashes. One thread per
    # process + N processes = clean 1:1 core mapping.
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_THREAD_LIMIT",
    ):
        os.environ.setdefault(v, "1")
    sys.path.insert(0, PARENT)
    import cv2
    import scoring_estimator as se

    cv2.setNumThreads(1)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    try:
        summ = se.run(
            video_path,
            outdir,
            start_frame=0,
            max_frames=None,
            ocr_every=ocr_every,
            field_mid=field_mid,
            k_slots=k_slots,
            write_video=write_video,
            crop=crop,
            frame_stride=stride,
        )
        dt = time.time() - t0
        sc = summ.get("alliance_scored_from_board", {})
        return (
            os.path.basename(video_path),
            "ok",
            f"blue={sc.get('blue')} red={sc.get('red')} ({dt:.0f}s)",
        )
    except Exception as e:
        return (
            os.path.basename(video_path),
            "FAIL",
            f"{e.__class__.__name__}: {e}\n{traceback.format_exc(limit=2)}",
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--workers", type=int, default=WORKERS, help="parallel processes (0 = min(cpu, 12))"
    )
    p.add_argument("--ocr-every", type=int, default=OCR_EVERY)
    p.add_argument("--slots", type=int, default=SLOTS, help="robots per alliance")
    p.add_argument(
        "--video",
        action="store_true",
        default=WRITE_VIDEO,
        help="also write annotated.mp4 per match (much slower)",
    )
    p.add_argument("--field-mid", type=int, help="x px splitting blue|red halves")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--only", nargs="*", default=ONLY, help="only these match keys")
    p.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help="process every Nth frame (2 = ~2x faster on 60fps)",
    )
    p.add_argument(
        "--force", action="store_true", default=FORCE, help="re-run even if estimate exists"
    )
    args = p.parse_args()

    workers = args.workers or min(os.cpu_count() or 4, 12)
    crop = tuple(args.crop) if args.crop else None

    vids = sorted(glob.glob(os.path.join(VIDEOS, "*.mp4")))
    if args.only:
        want = set(args.only)
        vids = [v for v in vids if os.path.splitext(os.path.basename(v))[0] in want]
    if not vids:
        raise SystemExit(f"No videos in {VIDEOS} (run download_matches.py first).")

    jobs = []
    for v in vids:
        mk = os.path.splitext(os.path.basename(v))[0]
        outdir = os.path.join(RESULTS, mk)
        if not args.force and os.path.exists(os.path.join(outdir, "estimate.json")):
            continue
        jobs.append(
            (v, outdir, args.ocr_every, args.slots, args.video, crop, args.field_mid, args.stride)
        )

    print(
        f"{len(vids)} videos, {len(jobs)} to run, {workers} workers "
        f"(skipping {len(vids) - len(jobs)} already done)"
    )
    if not jobs:
        return
    ok = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status, info = fut.result()
            print(f"  [{i}/{len(jobs)}] {name:22s} {status}: {info}")
            if status == "ok":
                ok += 1
            else:
                fail += 1
    print(
        f"\nDone: {ok} ok, {fail} failed in {time.time() - t0:.0f}s. "
        f"Estimates in {RESULTS}/<match>/estimate.json"
    )
    print("Next: python compare_tba.py")


if __name__ == "__main__":
    main()
