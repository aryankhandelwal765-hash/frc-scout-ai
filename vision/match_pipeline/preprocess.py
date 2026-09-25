"""
Clean match videos before the vision pipeline:
  - TRIM time: cut the dead pre-match / post-match seconds (auto-detected from when
    the scoreboard ball count goes live and stops climbing; manual override too).
  - CROP space: drop the lower stitched 2nd camera angle + sponsor banner, keep the
    scoreboard (top) + the top field. Removes a big source of motion noise.
  - TRANSCODE av1 -> h264, so OpenCV decodes faster downstream.

Non-destructive: the original moves to videos_raw/<match>.mp4 the first time; the
cleaned file replaces videos/<match>.mp4, so every other script uses it unchanged.
Coordinate systems are preserved (crop is from the TOP, y=0), so the detector CROP
and scoreboard OCR boxes still line up.

PyCharm: set ONLY / PAD below, click Run.
"""

import glob
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import pytesseract

# Tesseract binary. Defaults to whatever is on PATH; override with
# TESSERACT_CMD when it lives somewhere else (e.g. /opt/homebrew/bin/tesseract).
_TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
from boxes import TIMER_BOX_FRAC   # centre match timer "M:SS", as frame fractions

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
VIDEOS = os.path.join(HERE, "videos")
RAW = os.path.join(HERE, "videos_raw")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
ONLY = []              # e.g. ["2026hop_qm1"]; [] = all videos
# --- multi-view (fusion) ------------------------------------------------------
# The broadcast is a stitch: a top overhead view plus TWO low reverse views (one
# behind each hub tower) side by side underneath. At 1080p the reverse views are
# sharp enough to READ BUMPER NUMBERS, and they see behind the towers where the top
# view is blind -- so we keep all three as separate streams instead of cropping the
# lower band away. Rects are FRACTIONS of frame size, so they hold at any source
# resolution. Verified on 1920x1080.
MULTIVIEW = True
VIEW_RECTS = {                       # name -> (x, y, w, h) as fractions
    "top":  (0.000, 0.000, 1.000, 0.661),
    "revL": (0.004, 0.674, 0.456, 0.236),
    "revR": (0.539, 0.674, 0.457, 0.236),
}
KEEP_BOTTOM = 476      # legacy single-view crop (used when MULTIVIEW = False)
PAD = 2.0              # seconds of pad kept on each side of the detected match window
MANUAL = None          # override auto-trim: (start_sec, end_sec); None = auto-detect
SAMPLE_EVERY = 2.0     # seconds between timer OCR samples for match-window detect
WORKERS = 8            # parallel videos (OCR + ffmpeg are CPU-bound)
DELETE_RAW = True      # delete the original after a verified-good clean (no backup kept)
FORCE = False          # re-clean even if already done
# ===========================================================================


def read_timer(frame, box=None):
    """OCR the center match timer -> seconds remaining, or None. Rejects garbage
    (needs a colon, minutes<=5, seconds<60)."""
    if box is None:
        h, w = frame.shape[:2]
        fx1, fy1, fx2, fy2 = TIMER_BOX_FRAC
        box = (int(fx1 * w), int(fy1 * h), int(fx2 * w), int(fy2 * h))
    x1, y1, x2, y2 = box
    c = frame[y1:y2, x1:x2]
    big = cv2.resize(c, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    txt = pytesseract.image_to_string(
        th, config="--psm 7 -c tessedit_char_whitelist=0123456789:").strip()
    if ":" not in txt:
        return None
    a, b = txt.split(":")[:2]
    if not (a.isdigit() and b.isdigit()):
        return None
    m, s = int(a), int(b)
    if m > 5 or s >= 60:
        return None
    return m * 60 + s


def match_window(video, pad=PAD, sample_every=SAMPLE_EVERY):
    """Detect [start,end] seconds from the TIMER only (fast -- one OCR per sample).

    START: first frame the timer counts DOWN (match goes live; static pre-match
    timer doesn't decrease).
    END: the TELEOP zero -- the timer hits 0 AFTER it has been past ~2:00. It also
    hits 0 at the end of the ~15s auto, so we only take a zero that follows a
    reading >= 2:00 (the teleop countdown), never the auto zero.

    Seeks to only the sampled frames (decodes ~100, not every frame) and OCRs one
    tiny timer crop each -- so the whole scan is a few seconds.
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(sample_every * fps))
    prev = None
    first_dec = None
    timer_reads = []                  # (frame, seconds_remaining)
    f = 0
    while f < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        sec = read_timer(frame)
        if sec is not None:
            timer_reads.append((f, sec))
            if prev is not None and sec < prev and first_dec is None:
                first_dec = f
            prev = sec
        f += step
    cap.release()

    if first_dec is not None:
        start = first_dec / fps - pad
    elif timer_reads:
        start = timer_reads[0][0] / fps - pad
    else:
        start = 0.0

    end = None
    teleop = [r for r in timer_reads if r[1] >= 115]      # >= ~1:55 => teleop
    if teleop:
        peak_frame = max(teleop, key=lambda r: r[1])[0]
        after = [r for r in timer_reads if r[0] > peak_frame]
        zeros = [r for r in after if r[1] <= 1]
        if zeros:
            end = min(zeros, key=lambda r: r[0])[0] / fps + 1.0   # buzzer + tiny pad
        elif after:
            end = max(after, key=lambda r: r[0])[0] / fps + 2.0   # timer vanished early
    if end is None:
        end = (total / fps) if total > 0 else 0.0             # fallback: whole video

    return max(0.0, start), min(total / fps if total > 0 else end, end)


def _worker(args):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    sys.path.insert(0, HERE)
    sys.path.insert(0, PARENT)
    cv2.setNumThreads(1)
    return clean_one(*args)


def _height(path):
    c = cv2.VideoCapture(path)
    h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    c.release()
    return h


def _view_dir(view):
    return VIDEOS if view == "top" else os.path.join(VIDEOS, view)


def clean_one(match, keep_bottom, pad, manual, force, delete_raw=DELETE_RAW):
    cur = os.path.join(VIDEOS, f"{match}.mp4")
    raw = os.path.join(RAW, f"{match}.mp4")
    src = cur if os.path.exists(cur) else (raw if os.path.exists(raw) else None)
    if src is None:
        return match, "FAIL: no video"

    if MULTIVIEW:
        return _clean_multiview(match, src, cur, raw, pad, manual, force, delete_raw)

    # already cleaned? cleaned videos are cropped to keep_bottom; originals are taller
    if not force and 0 < _height(src) <= keep_bottom:
        return match, "skip (already cleaned)"

    start, end = manual if manual else match_window(src, pad)
    dur = max(0.1, end - start)
    tmp = os.path.join(VIDEOS, f".{match}.clean.tmp.mp4")
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", src,
           "-vf", f"crop=iw:{keep_bottom}:0:0",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an",
           "-threads", "3",              # cap per-job threads so parallel jobs don't thrash
           "-loglevel", "error", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    # verify the clean is good BEFORE we touch/delete the original
    good = r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0
    if not good:
        if os.path.exists(tmp):
            os.remove(tmp)
        tail = (r.stderr or "").strip().splitlines()
        return match, "FAIL: " + (tail[-1] if tail else "ffmpeg error")

    if delete_raw:
        os.replace(tmp, cur)                      # overwrite original -> it's gone
        if os.path.exists(raw):
            os.remove(raw)
        note = " (orig deleted)"
    else:
        os.makedirs(RAW, exist_ok=True)
        if os.path.exists(cur) and not os.path.exists(raw):
            shutil.copy(cur, raw)                 # keep a backup once
        os.replace(tmp, cur)
        note = ""
    sz = os.path.getsize(cur) // (1024 * 1024)
    return match, f"ok  trim[{start:.0f}-{end:.0f}s] crop->{keep_bottom}px  ({sz} MB){note}"


def _clean_multiview(match, src, cur, raw, pad, manual, force, delete_raw):
    """Trim once, then emit one stream per view: videos/<match>.mp4 (top) and
    videos/<view>/<match>.mp4 for each reverse view. All share the same time
    window, so frame N lines up across views -- that's what makes fusion possible."""
    outs = {v: os.path.join(_view_dir(v), f"{match}.mp4") for v in VIEW_RECTS}
    if not force and all(os.path.exists(p) and os.path.getsize(p) > 0
                         for v, p in outs.items() if v != "top"):
        return match, "skip (already split)"

    cap = cv2.VideoCapture(src)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not W or not H:
        return match, "FAIL: unreadable source"

    start, end = manual if manual else match_window(src, pad)
    dur = max(0.1, end - start)

    made = []
    for view, (fx, fy, fw, fh) in VIEW_RECTS.items():
        x, y = int(fx * W), int(fy * H)
        w, h = int(fw * W) // 2 * 2, int(fh * H) // 2 * 2      # even dims for h264
        os.makedirs(_view_dir(view), exist_ok=True)
        tmp = os.path.join(_view_dir(view), f".{match}.{view}.tmp.mp4")
        cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", src,
               "-vf", f"crop={w}:{h}:{x}:{y}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an",
               "-threads", "3", "-loglevel", "error", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            for p in made:                        # roll back partial split
                if os.path.exists(p):
                    os.remove(p)
            if os.path.exists(tmp):
                os.remove(tmp)
            tail = (r.stderr or "").strip().splitlines()
            return match, f"FAIL[{view}]: " + (tail[-1] if tail else "ffmpeg error")
        made.append(tmp)

    # every view encoded OK -> commit, then (optionally) drop the original
    for view in VIEW_RECTS:
        tmp = os.path.join(_view_dir(view), f".{match}.{view}.tmp.mp4")
        os.replace(tmp, outs[view])
    note = ""
    if delete_raw and os.path.exists(raw):
        os.remove(raw)
        note = " (orig deleted)"
    mb = sum(os.path.getsize(p) for p in outs.values()) // (1024 * 1024)
    return match, (f"ok  trim[{start:.0f}-{end:.0f}s] {len(VIEW_RECTS)} views "
                   f"@{W}x{H}  ({mb} MB){note}")


def main():
    vids = sorted(glob.glob(os.path.join(VIDEOS, "*.mp4")))
    matches = [os.path.splitext(os.path.basename(v))[0] for v in vids]
    if ONLY:
        matches = [m for m in matches if m in set(ONLY)]
    # include ones already moved to raw (so re-run/force works)
    for r in glob.glob(os.path.join(RAW, "*.mp4")):
        m = os.path.splitext(os.path.basename(r))[0]
        if m not in matches and (not ONLY or m in set(ONLY)):
            matches.append(m)
    matches = sorted(set(matches))
    print(f"{len(matches)} videos to clean, {WORKERS} workers "
          f"(keep top {KEEP_BOTTOM}px, timer-trimmed)")
    ok = fail = skip = 0
    jobs = [(m, KEEP_BOTTOM, PAD, MANUAL, FORCE) for m in matches]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_worker, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            m = futs[fut]
            _, status = fut.result()
            print(f"  [{i}/{len(jobs)}] {m:16s} {status}")
            ok += status.startswith("ok")
            skip += status.startswith("skip")
            fail += status.startswith("FAIL")
    print(f"\nDone: {ok} cleaned, {skip} skipped, {fail} failed. "
          f"Originals in {RAW}/")


if __name__ == "__main__":
    main()
