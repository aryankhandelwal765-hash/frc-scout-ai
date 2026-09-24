"""
Detector-based per-robot ball-scoring estimator.

This replaces the motion-hotspot proxy with the trained YOLO robot detector, which
fixes the two flaws that made the proxy "heavily wrong":
  - boxes are on ACTUAL ROBOTS (not ball piles / people), and
  - alliance comes from the robot's BUMPER COLOR (the detector class), not a
    bogus field-half split.

Per frame:
  1. Detect robots (blue_robot / red_robot) on the cropped top-field view.
  2. Track each robot across frames -> persistent per-robot IDs, per alliance.
  3. Measure each robot's scoring activity = moving-ball motion around its box
     (a ball leaving/around the robot).
  4. OCR the scoreboard for the authoritative per-alliance ball totals.
  5. Apportion each alliance's real scored total across ITS robots by activity.

Per-alliance totals are exact (scoreboard). The per-robot split is an estimate,
now built on real robots + real alliance. Team-number identity is still not
recoverable at this resolution (robots are unnamed A/B/C within an alliance).

    python detector_estimator.py --input test_clip.mp4 \
        --weights runs/detect/frc_robots-2/weights/best.pt --outdir out_detector
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# The classic-CV modules this builds on live in vision/classic.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "classic"))

from scoring_estimator import (SCOREBOARD, YELLOW_LO, YELLOW_HI,
                               ocr_ball_count, moving_ball_mask)

CROP = (0, 100, 1280, 470)     # MUST match the crop the detector was trained on
BLUE_BGR = (255, 120, 0)
RED_BGR = (0, 80, 255)
ACT_MARGIN = 25                # px to expand a robot box when summing ball motion
ROBOT_MAX_DIST = 110           # px a robot centroid may jump between frames
ROBOT_GRACE = 45               # frames a robot track survives detection misses


def run(input_path, weights, outdir, start_frame, max_frames, ocr_every, conf,
        imgsz, write_video, k_robots=3, tracker="bytetrack.yaml"):
    os.makedirs(outdir, exist_ok=True)
    model = YOLO(weights)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    x1, y1, x2, y2 = CROP
    cw, ch = x2 - x1, y2 - y1

    writer = None
    if write_video:
        writer = cv2.VideoWriter(os.path.join(outdir, "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))

    # ByteTrack (via ultralytics) maintains stable robot IDs across occlusions --
    # far better than a naive centroid tracker over a full match.
    activity = defaultdict(int)     # tid -> cumulative ball-motion
    frames_seen = defaultdict(int)  # tid -> frames visible
    pos_sum = defaultdict(lambda: np.zeros(2))
    alliance_votes = defaultdict(lambda: [0, 0])  # tid -> [blue, red] class counts

    board = {"blue": {"first": None, "max": 0, "last": None},
             "red":  {"first": None, "max": 0, "last": None}}
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
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        mv = moving_ball_mask(gray, prev_gray, yellow, (0, 0, cw, ch))

        # 1+2. detect + track robots with ByteTrack (stable IDs across occlusion)
        res = model.track(crop, conf=conf, imgsz=imgsz, agnostic_nms=True,
                          tracker=tracker, persist=True, verbose=False)[0]
        # 3. accumulate per-robot ball-motion activity, keyed by stable track id
        live = []  # (tid, alliance, bbox)
        if res.boxes is not None and res.boxes.id is not None:
            for b in res.boxes:
                tid = int(b.id)
                cls = int(b.cls)
                alliance_votes[tid][cls] += 1
                bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0]]
                bw, bh = bx2 - bx1, by2 - by1
                ax1 = max(0, bx1 - ACT_MARGIN); ay1 = max(0, by1 - ACT_MARGIN)
                ax2 = min(cw, bx2 + ACT_MARGIN); ay2 = min(ch, by2 + ACT_MARGIN)
                activity[tid] += int(cv2.countNonZero(mv[ay1:ay2, ax1:ax2]))
                frames_seen[tid] += 1
                pos_sum[tid] += np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0])
                alliance = "blue" if cls == 0 else "red"
                live.append((tid, alliance, (bx1, by1, bw, bh)))

        # 4. OCR the scoreboard (full frame; it's above the crop)
        if processed % ocr_every == 0:
            b = ocr_ball_count(frame, SCOREBOARD["blue"])
            r = ocr_ball_count(frame, SCOREBOARD["red"])
            for alliance, val in (("blue", b), ("red", r)):
                if val is None or val > 1500:
                    continue
                bd = board[alliance]
                if bd["max"] and val < bd["max"] - 5:
                    continue
                if bd["first"] is None:
                    bd["first"] = val
                bd["last"] = val
                bd["max"] = max(bd["max"], val)
            board_series.append((frame_idx, b, r))

        if writer is not None:
            _draw(crop, live, activity, board, frame_idx)
            writer.write(crop)

        prev_gray = gray
        frame_idx += 1
        processed += 1

    cap.release()
    if writer is not None:
        writer.release()

    return _summarize(outdir, board, board_series, activity, frames_seen, pos_sum,
                      alliance_votes, processed, fps, k_robots)


def _draw(crop, live, activity, board, frame_idx):
    for tid, alliance, bbox in live:
        col = BLUE_BGR if alliance == "blue" else RED_BGR
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(crop, (x, y), (x + w, y + h), col, 2)
        cv2.putText(crop, f"{alliance[0].upper()}#{tid} act:{activity[tid]}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    txt = f"BOARD blue:{board['blue']['last']}  red:{board['red']['last']}  f={frame_idx}"
    cv2.rectangle(crop, (6, 6), (500, 34), (0, 0, 0), -1)
    cv2.putText(crop, txt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def _summarize(outdir, board, series, activity, frames_seen, pos_sum, alliance_votes,
               processed, fps, k_robots):
    scored = {a: (board[a]["max"] - board[a]["first"]) if board[a]["first"] is not None
              else 0 for a in ("blue", "red")}
    all_tracks = []
    for tid, act in activity.items():
        avg = pos_sum[tid] / max(1, frames_seen[tid])
        alliance = "blue" if alliance_votes[tid][0] >= alliance_votes[tid][1] else "red"
        all_tracks.append({"alliance": alliance, "id": tid, "activity": act,
                           "frames_seen": frames_seen[tid],
                           "avg_pos": [round(float(avg[0]), 1), round(float(avg[1]), 1)]})
    # collapse detector-flicker to the real K robots: keep the K most-persistent
    # tracks per alliance, then apportion that alliance's scored total among them.
    robots = []
    for alliance in ("blue", "red"):
        grp = sorted([r for r in all_tracks if r["alliance"] == alliance],
                     key=lambda r: -r["frames_seen"])[:k_robots]
        tot = sum(r["activity"] for r in grp) or 1
        for r in grp:
            r["est_balls_scored"] = round(scored[alliance] * r["activity"] / tot, 1)
        robots.extend(grp)

    robots.sort(key=lambda r: (r["alliance"], -r["est_balls_scored"]))
    summary = {"frames_processed": processed,
               "duration_sec": round(processed / fps, 2) if fps else None,
               "alliance_scored_from_board": scored,
               "board_first_max": {a: [board[a]["first"], board[a]["max"]] for a in board},
               "robots": robots}
    with open(os.path.join(outdir, "estimate.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(outdir, "board_series.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "blue_balls", "red_balls"]); w.writerows(series)
    _print(summary)
    return summary


def _print(s):
    print("\n" + "=" * 58)
    print("  DETECTOR-BASED PER-ROBOT BALL-SCORING ESTIMATE")
    print("=" * 58)
    print(f"  frames: {s['frames_processed']} ({s['duration_sec']}s)")
    sc = s["alliance_scored_from_board"]
    print(f"  balls scored (scoreboard truth): BLUE {sc['blue']}  RED {sc['red']}")
    for alliance in ("blue", "red"):
        grp = sorted([r for r in s["robots"] if r["alliance"] == alliance],
                     key=lambda r: -r["est_balls_scored"])
        letters = "ABCDEFGH"
        print(f"  --- {alliance.upper()} ({len(grp)} tracked robots) ---")
        for j, r in enumerate(grp):
            print(f"    {alliance:4s} robot {letters[j]:2s} ~{r['est_balls_scored']:>6} balls"
                  f"   (activity {r['activity']}, seen {r['frames_seen']}f, pos {r['avg_pos']})")
    print("=" * 58)
    print("  per-alliance = exact (scoreboard); per-robot = estimate on real robots.\n")


def main():
    p = argparse.ArgumentParser(description="Per-robot ball scoring via trained YOLO detector.")
    p.add_argument("--input", required=True)
    p.add_argument("--weights", default="runs/detect/frc_robots-2/weights/best.pt")
    p.add_argument("--outdir", default="out_detector")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--ocr-every", type=int, default=10)
    p.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--robots-per-alliance", type=int, default=3,
                   help="collapse to the K most-persistent robot tracks per alliance")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="tracker config (e.g. bytetrack_frc.yaml for long occlusion buffer)")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    run(args.input, args.weights, args.outdir, args.start_frame, args.max_frames,
        args.ocr_every, args.conf, args.imgsz, write_video=not args.no_video,
        k_robots=args.robots_per_alliance, tracker=args.tracker)


if __name__ == "__main__":
    main()
