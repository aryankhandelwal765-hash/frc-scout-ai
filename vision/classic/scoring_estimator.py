"""
Rate-apportionment scoring estimator (per-robot).

Estimate how many balls EACH robot scores in a match, on bulk-dump footage where
you cannot count or ID individual balls.

Method (the user's idea):
  1. RATE, not count. Each frame, find balls that are MOVING (a dumping/scoring
     event stirs balls; static floor balls are ignored). Moving-ball motion near
     a robot = that robot's release activity.
  2. ROBOT SLOTS, not detections. Robots aren't detectable on this footage, so we
     keep exactly K "slots" per alliance (K = robots per alliance, default 3).
     Each slot is a persistent activity center that follows a robot's dumping,
     seeded lazily and spread apart, EMA-tracked. This collapses the thousands of
     fragmentary motion blobs into K stable per-robot signals.
  3. TRUTH from the scoreboard. OCR the on-screen per-alliance ball totals -- the
     authoritative count of balls actually scored.
  4. APPORTION. Split each alliance's real scored total among its K slots in
     proportion to their cumulative activity share.

For per-MATCH totals the scoreboard delay is irrelevant -- we only need start/end
totals and cumulative activity shares.

ESTIMATOR, not ground truth. Per-ALLIANCE totals are exact (scoreboard). The
per-slot split is a heuristic: it's only as separable as the robots are in space.
When all activity is one churning pile, the split across slots is unreliable, and
slots are unnamed (A/B/C) -- bumper team numbers aren't readable at this scale.
"""

import argparse
import csv
import json
import os

import cv2
import numpy as np
import pytesseract

# Tesseract binary. Defaults to whatever is on PATH; override with
# TESSERACT_CMD when it lives somewhere else (e.g. /opt/homebrew/bin/tesseract).
_TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

# ---------------------------------------------------------------------------
# Config -- tuned for the 1280x720 "FIRST AGE" broadcast layout. Adjust per feed.
# ---------------------------------------------------------------------------
SCOREBOARD = {
    # alliance -> (x1, x2, y1, y2) box around the "<scored> / <total>" ball count
    "blue": (60, 160, 38, 86),
    "red": (1160, 1270, 38, 86),
}
# The boxes above are pixels for a 1280x720 broadcast. Source is 1080p now, so keep
# them as FRACTIONS of frame width/height and resolve per frame -- otherwise the OCR
# samples empty pixels and every read comes back None.
# Measured on the pre-split top view (1920x712): these read '42/360' and '135/360'
# on a frame whose board showed exactly that.
SCOREBOARD_FRAC = {
    "blue": (0.0156, 0.1250, 0.0843, 0.1826),
    "red": (0.8750, 0.9896, 0.0843, 0.1826),
}


def scoreboard_box(frame, alliance):
    h, w = frame.shape[:2]
    fx1, fx2, fy1, fy2 = SCOREBOARD_FRAC[alliance]
    return int(fx1 * w), int(fx2 * w), int(fy1 * h), int(fy2 * h)


YELLOW_LO = np.array([18, 55, 80])
YELLOW_HI = np.array([45, 255, 255])

# Robust scoreboard tracking. The counter climbs monotonically. A smooth step
# (<= SOFT above last good) is trusted immediately. A big jump up is EITHER an OCR
# spike (garbage "1150") OR a real catch-up after OCR dropped some frames -- so we
# don't trust it until a SECOND nearby read confirms it (spikes are one-off and
# never confirm; real climbs are sustained and do). Don't seed from a big first
# read either (match starts near 0). Kills spikes without stranding the count.
SOFT = 40
FIRST_MAX = 80

FIELD_TOP = 100  # ignore motion above this row (scoreboard/crowd strip)
FIELD_BOTTOM = 660  # ignore motion below this row (sponsor banner)
MOTION_THRESH = 25  # grayscale abs-diff to call a pixel "moving"
AGENT_DILATE = 25  # px; merge nearby ball-motion into one cluster
AGENT_MIN_ACTIVITY = 40  # min moving-ball px in a cluster to count
SLOT_SEED_SEP = 160  # px; a cluster this far from all slots seeds a new one
SLOT_EMA = 0.08  # how fast a slot follows its robot (0=frozen,1=jumpy)

BLUE = (255, 120, 0)
RED = (0, 80, 255)


class SlotTracker:
    """Exactly K persistent activity slots (robot proxies) for one alliance.

    Slots are seeded lazily: a cluster far from every existing slot creates a new
    one (until K exist), otherwise it feeds the nearest slot and nudges its
    position. This keeps K stable per-robot signals instead of thousands of
    fragmentary tracks."""

    def __init__(self, k, seed_sep=SLOT_SEED_SEP, ema=SLOT_EMA):
        self.k = k
        self.seed_sep = seed_sep
        self.ema = ema
        self.slots = []  # each: {x, y, activity, hits}

    def _nearest(self, cx, cy):
        best, bd = None, float("inf")
        for i, s in enumerate(self.slots):
            d = ((cx - s["x"]) ** 2 + (cy - s["y"]) ** 2) ** 0.5
            if d < bd:
                bd, best = d, i
        return best, bd

    def update(self, clusters):
        # biggest clusters claim/anchor slots first
        for c in sorted(clusters, key=lambda c: -c["activity"]):
            cx, cy = c["centroid"]
            act = c["activity"]
            i, d = self._nearest(cx, cy)
            if i is None or (d > self.seed_sep and len(self.slots) < self.k):
                self.slots.append({"x": cx, "y": cy, "activity": act, "hits": 1})
            else:
                s = self.slots[i]
                s["x"] += self.ema * (cx - s["x"])
                s["y"] += self.ema * (cy - s["y"])
                s["activity"] += act
                s["hits"] += 1


def ocr_ball_count(frame, box):
    """Read the numerator ('balls scored so far') from a scoreboard box.
    `box` may be an (x1,x2,y1,y2) tuple or an alliance name ('blue'/'red'), in
    which case it is resolved against this frame's actual size."""
    if isinstance(box, str):
        box = scoreboard_box(frame, box)
    x1, x2, y1, y2 = box
    crop = frame[y1:y2, x1:x2]
    big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    txt = pytesseract.image_to_string(
        th, config="--psm 7 -c tessedit_char_whitelist=0123456789/"
    ).strip()
    # The board reads "<scored> / <total>". Require the slash: a bare number is
    # almost always the TOTAL ("360") read alone with the numerator missed, and
    # since it repeats every frame it would sail past the confirmation check and
    # lock the count to 360. No slash -> unreliable read -> skip it.
    if "/" not in txt:
        return None
    head = txt.split("/")[0]
    digits = "".join(c for c in head if c.isdigit())
    return int(digits) if digits else None


def moving_ball_mask(frame_gray, prev_gray, yellow, roi):
    """roi = (x1, y1, x2, y2): only motion inside this rectangle is kept. This is
    how we exclude the stitched second camera angle, crowd, scoreboard, banner."""
    if prev_gray is None:
        return np.zeros_like(yellow)
    motion = cv2.absdiff(frame_gray, prev_gray)
    motion = cv2.threshold(motion, MOTION_THRESH, 255, cv2.THRESH_BINARY)[1]
    motion = cv2.dilate(motion, np.ones((3, 3), np.uint8))
    mv = cv2.bitwise_and(yellow, motion)
    x1, y1, x2, y2 = roi
    keep = np.zeros_like(mv)
    keep[y1:y2, x1:x2] = mv[y1:y2, x1:x2]
    return keep


def find_clusters(mv_mask):
    """Cluster moving-ball pixels into blobs with centroid + activity."""
    dil = cv2.dilate(mv_mask, np.ones((AGENT_DILATE, AGENT_DILATE), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dil, connectivity=8)
    out = []
    for i in range(1, n):
        activity = int(
            cv2.countNonZero(cv2.bitwise_and(mv_mask, (labels == i).astype(np.uint8) * 255))
        )
        if activity < AGENT_MIN_ACTIVITY:
            continue
        out.append({"centroid": (float(cents[i][0]), float(cents[i][1])), "activity": activity})
    return out


def pick_crop(input_path):
    """Show the first frame; drag a box around the region to KEEP (the top field).
    Needs a display -- run from a terminal, not the PyCharm run window. Returns
    (x1, y1, x2, y2) and prints it so you can reuse it via --crop."""
    cap = cv2.VideoCapture(input_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Could not read a frame from {input_path}")
    print("\nDrag a box around the part to KEEP (e.g. the top overhead field).")
    print("Press ENTER/SPACE to confirm, ESC to cancel.\n")
    x, y, w, h = cv2.selectROI(
        "Select region to KEEP (ENTER=ok, ESC=cancel)", frame, showCrosshair=False
    )
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        raise SystemExit("No region selected.")
    crop = (int(x), int(y), int(x + w), int(y + h))
    print(f"Selected crop: --crop {crop[0]} {crop[1]} {crop[2]} {crop[3]}\n")
    return crop


def run(
    input_path,
    outdir,
    start_frame,
    max_frames,
    ocr_every,
    field_mid,
    k_slots,
    write_video,
    crop=None,
    frame_stride=1,
):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if field_mid is None:
        field_mid = W // 2

    # crop = region to analyze AND export. Default: full frame minus the fixed
    # field band (backward compatible). OCR always reads the full-frame scoreboard.
    if crop is None:
        roi = (0, FIELD_TOP, W, FIELD_BOTTOM)
        cx1, cy1, cx2, cy2 = 0, 0, W, H  # export whole frame
    else:
        roi = crop
        cx1, cy1, cx2, cy2 = crop  # export just the cropped region

    writer = None
    if write_video:
        writer = cv2.VideoWriter(
            os.path.join(outdir, "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (cx2 - cx1, cy2 - cy1),
        )

    slots = {"blue": SlotTracker(k_slots), "red": SlotTracker(k_slots)}
    board = {
        "blue": {"first": None, "last": None, "max": 0, "last_good": None, "cand": None},
        "red": {"first": None, "last": None, "max": 0, "last_good": None, "cand": None},
    }
    board_series = []

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_gray = None
    frame_idx = start_frame
    processed = 0

    while True:
        if max_frames is not None and processed >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        # frame_stride>1: skip the heavy per-frame work on off-stride frames (60fps
        # footage doesn't need every frame). prev_gray stays the last PROCESSED
        # frame, so motion is measured across the stride. Per-alliance OCR total is
        # unaffected; per-robot split is marginally noisier.
        if frame_stride > 1 and (frame_idx - start_frame) % frame_stride != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)

        mv = moving_ball_mask(gray, prev_gray, yellow, roi)
        clusters = find_clusters(mv)
        # split clusters by field side -> alliance, feed each alliance's slots
        blue_c = [c for c in clusters if c["centroid"][0] < field_mid]
        red_c = [c for c in clusters if c["centroid"][0] >= field_mid]
        slots["blue"].update(blue_c)
        slots["red"].update(red_c)

        if processed % ocr_every == 0:
            b = ocr_ball_count(frame, "blue")
            r = ocr_ball_count(frame, "red")
            for alliance, val in (("blue", b), ("red", r)):
                if val is None or val > 1500:
                    continue
                bd = board[alliance]
                lg = bd["last_good"]
                if lg is None:
                    # match starts near 0, so a big first read is an OCR spike.
                    if val <= FIRST_MAX:
                        bd["first"] = bd["last"] = bd["last_good"] = val
                        bd["max"] = val
                elif val < lg - 5:
                    continue  # backward misread -> ignore
                elif val <= lg + SOFT:
                    # smooth monotonic step -> trust immediately
                    bd["last"] = bd["last_good"] = val
                    bd["max"] = max(bd["max"], val)
                    bd["cand"] = None
                elif bd["cand"] is not None and abs(val - bd["cand"]) <= SOFT:
                    # big jump, but a second nearby read confirms it -> real catch-up
                    nv = max(val, bd["cand"])
                    bd["last"] = bd["last_good"] = nv
                    bd["max"] = max(bd["max"], nv)
                    bd["cand"] = None
                else:
                    bd["cand"] = val  # hold one suspect; spikes never confirm
            board_series.append((frame_idx, b, r))

        if writer is not None:
            _draw_slots(frame, slots, field_mid, roi)
            out = frame[cy1:cy2, cx1:cx2].copy()
            _draw_board_text(out, board, frame_idx)
            writer.write(out)

        prev_gray = gray
        frame_idx += 1
        processed += 1

    cap.release()
    if writer is not None:
        writer.release()
    return _summarize(outdir, board, board_series, slots, processed, fps)


def _draw_slots(frame, slots, mid, roi):
    """Draw the alliance split line + slot circles on the full frame (so their
    positions stay aligned when the frame is later cropped)."""
    x1, y1, x2, y2 = roi
    cv2.line(frame, (mid, y1), (mid, y2), (200, 200, 200), 1)
    for alliance, col in (("blue", BLUE), ("red", RED)):
        for i, s in enumerate(slots[alliance].slots):
            x, y = int(s["x"]), int(s["y"])
            r = int(np.clip(np.sqrt(s["activity"]) / 12, 12, 90))
            cv2.circle(frame, (x, y), r, col, 3)
            tag = "B" if alliance == "blue" else "R"
            cv2.putText(
                frame,
                f"{tag}{i} act:{s['activity']}",
                (x - 40, y - r - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                2,
            )


def _draw_board_text(img, board, frame_idx):
    """Overlay the OCR'd board readout at the top-left of the (possibly cropped)
    output, so it stays visible even when the real scoreboard is cropped away."""
    txt = f"BOARD blue:{board['blue']['last']}  red:{board['red']['last']}  f={frame_idx}"
    cv2.rectangle(img, (8, 8), (520, 40), (0, 0, 0), -1)
    cv2.putText(img, txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _summarize(outdir, board, series, slots, processed, fps):
    scored = {
        a: (board[a]["max"] - board[a]["first"]) if board[a]["first"] is not None else 0
        for a in ("blue", "red")
    }
    robots = []
    for alliance in ("blue", "red"):
        st = slots[alliance].slots
        total_act = sum(s["activity"] for s in st) or 1
        for i, s in enumerate(st):
            robots.append(
                {
                    "alliance": alliance,
                    "slot": i,
                    "activity": s["activity"],
                    "hits": s["hits"],
                    "pos": [round(s["x"], 1), round(s["y"], 1)],
                    "est_balls_scored": round(scored[alliance] * s["activity"] / total_act, 1),
                }
            )
    summary = {
        "frames_processed": processed,
        "duration_sec": round(processed / fps, 2) if fps else None,
        "alliance_scored_from_board": scored,
        "board_first_last": {a: [board[a]["first"], board[a]["max"]] for a in board},
        "robots": robots,
    }
    with open(os.path.join(outdir, "estimate.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(outdir, "board_series.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "blue_balls", "red_balls"])
        w.writerows(series)
    _print(summary)
    return summary


def _print(s):
    print("\n" + "=" * 56)
    print("  PER-ROBOT BALL-SCORING ESTIMATE")
    print("=" * 56)
    print(f"  frames: {s['frames_processed']} ({s['duration_sec']}s)")
    sc = s["alliance_scored_from_board"]
    print(f"  balls scored (scoreboard truth): BLUE {sc['blue']}  RED {sc['red']}")
    print("  estimated balls per robot slot (unnamed A/B/C):")
    for alliance in ("blue", "red"):
        grp = sorted(
            [r for r in s["robots"] if r["alliance"] == alliance],
            key=lambda r: -r["est_balls_scored"],
        )
        letters = "ABCDEF"
        print(f"    --- {alliance.upper()} ---")
        for j, r in enumerate(grp):
            print(
                f"    {alliance:4s} robot {letters[j]}  ~{r['est_balls_scored']:>6} balls"
                f"   (activity {r['activity']}, pos {r['pos']})"
            )
    print("=" * 56)
    print("  per-alliance totals are exact (scoreboard);")
    print("  per-robot split is an activity-weighted ESTIMATE, unnamed slots.\n")


def main():
    p = argparse.ArgumentParser(
        description="Estimate balls scored per robot via "
        "K-slot release-rate apportionment + scoreboard OCR."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="out_estimate")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--ocr-every", type=int, default=10, help="OCR the board every N frames")
    p.add_argument("--field-mid", type=int, help="x px splitting blue|red field halves")
    p.add_argument("--slots", type=int, default=3, help="robots per alliance (K)")
    p.add_argument("--no-video", action="store_true", help="skip annotated video (faster)")
    p.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="analyze + export only this rectangle (drops the stitched 2nd angle)",
    )
    p.add_argument(
        "--pick-crop",
        action="store_true",
        help="drag the keep-region on the first frame, then run with it",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="process every Nth frame (2 = ~2x faster on 60fps footage)",
    )
    args = p.parse_args()

    crop = tuple(args.crop) if args.crop else None
    if args.pick_crop:
        crop = pick_crop(args.input)
    run(
        args.input,
        args.outdir,
        args.start_frame,
        args.max_frames,
        args.ocr_every,
        args.field_mid,
        args.slots,
        write_video=not args.no_video,
        crop=crop,
        frame_stride=args.stride,
    )


if __name__ == "__main__":
    main()
