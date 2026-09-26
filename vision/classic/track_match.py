"""
FRC Match Vision -- tracks robots AND the game-piece balls they score.

Pipeline (all classic CV, CPU-only, no training):
    frame -> color masks -> robot blobs + ball blobs
          -> per-alliance robot trackers + ball tracker (with grace period)
          -> possession (ball -> nearest robot) + scoring (ball vanishes in a goal zone)
          -> annotated video + CSVs + JSON summary

Quick start:
    # 1. define where the goals are (once per camera angle):
    python track_match.py --input clip.mp4 --pick-zones --zones-out zones.json
    # 2. run the tracker:
    python track_match.py --input clip.mp4 --zones zones.json --outdir out/

See README.md for grabbing footage and tuning.
"""

import argparse
import csv
import json
import os

import cv2

import config
from config import Params, GoalZone
import detection
from tracking import CentroidTracker
from scoring import ScoringEngine
import overlay


# ---------------------------------------------------------------------------
# Goal-zone loading / interactive picking
# ---------------------------------------------------------------------------
def load_zones(path):
    with open(path) as f:
        data = json.load(f)
    zones = []
    for z in data:
        zones.append(GoalZone(name=z["name"], alliance=z["alliance"], rect=tuple(z["rect"])))
    return zones


def save_zones(zones, path):
    with open(path, "w") as f:
        json.dump(
            [{"name": z.name, "alliance": z.alliance, "rect": list(z.rect)} for z in zones],
            f,
            indent=2,
        )
    print(f"Saved {len(zones)} zone(s) -> {path}")


def pick_zones_interactive(input_path):
    """Show the first frame and let the user drag rectangles for each goal.
    Needs a display (fine on your Mac). Assigns alliances round-robin via prompt."""
    cap = cv2.VideoCapture(input_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Could not read a frame from {input_path}")

    print("\nZone picker:")
    print("  - Drag a box around each GOAL / scoring region.")
    print("  - Press ENTER/SPACE after each box; ESC when done.\n")
    rects = cv2.selectROIs("Select goal zones (ENTER=add, ESC=done)", frame, showCrosshair=False)
    cv2.destroyAllWindows()

    zones = []
    for i, r in enumerate(rects):
        x, y, w, h = [int(v) for v in r]
        if w == 0 or h == 0:
            continue
        alliance = (
            input(f"Zone {i} alliance [red/blue/shared] (default shared): ").strip() or "shared"
        )
        name = input(f"Zone {i} name (default goal{i}): ").strip() or f"goal{i}"
        zones.append(GoalZone(name=name, alliance=alliance, rect=(x, y, w, h)))
    return zones


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------
def process_video(
    input_path,
    outdir,
    zones,
    ball_ranges,
    params,
    show=False,
    start_frame=0,
    max_frames=None,
    detect_robots=True,
):
    os.makedirs(outdir, exist_ok=True)
    out_video = os.path.join(outdir, "annotated.mp4")
    robots_csv_path = os.path.join(outdir, "robot_positions.csv")
    balls_csv_path = os.path.join(outdir, "ball_positions.csv")
    events_csv_path = os.path.join(outdir, "scoring_events.csv")
    summary_path = os.path.join(outdir, "summary.json")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    red_tracker = CentroidTracker(params.robot_max_distance, params.robot_grace_frames)
    blue_tracker = CentroidTracker(params.robot_max_distance, params.robot_grace_frames)
    ball_tracker = CentroidTracker(params.ball_max_distance, params.ball_grace_frames)
    engine = ScoringEngine(zones, params.possession_radius, params.possession_bbox_margin)

    robots_csv = open(robots_csv_path, "w", newline="")
    robots_w = csv.writer(robots_csv)
    robots_w.writerow(["frame", "time_sec", "alliance", "track_id", "x", "y", "w", "h", "visible"])
    balls_csv = open(balls_csv_path, "w", newline="")
    balls_w = csv.writer(balls_csv)
    balls_w.writerow(["frame", "time_sec", "ball_id", "x", "y", "w", "h", "visible", "possessor"])

    flash = None  # (event, frames_remaining)
    flash_frames = int(fps * 1.2)
    frame_idx = 0

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

    while True:
        if max_frames is not None and frame_idx - start_frame >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = frame_idx / fps

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- detect ---
        if detect_robots:
            red_mask = detection.get_color_mask(hsv, config.RED_BUMPER_RANGES, params.morph_kernel)
            blue_mask = detection.get_color_mask(
                hsv, config.BLUE_BUMPER_RANGES, params.morph_kernel
            )
            red_blobs = detection.find_robot_blobs(red_mask, params.robot_min_area)
            blue_blobs = detection.find_robot_blobs(blue_mask, params.robot_min_area)
        else:
            red_blobs, blue_blobs = [], []
        ball_mask = detection.get_ball_mask(hsv, ball_ranges, params.ball_close_kernel)
        ball_blobs = detection.find_ball_blobs(
            ball_mask, params.ball_min_area, params.ball_max_area, params.ball_min_circularity
        )

        # --- track ---
        red_tracks, _ = red_tracker.update(red_blobs)
        blue_tracks, _ = blue_tracker.update(blue_blobs)
        ball_tracks, deleted_balls = ball_tracker.update(ball_blobs)

        robots_by_alliance = {"red": red_tracks, "blue": blue_tracks}

        # --- possession + scoring ---
        engine.update_possession(ball_tracks, robots_by_alliance)
        new_events = engine.check_scores(deleted_balls, frame_idx, time_sec)
        if new_events:
            flash = (new_events[-1], flash_frames)

        # --- CSV rows ---
        for alliance, tracks in robots_by_alliance.items():
            for rid, t in tracks.items():
                x, y, w, h = [int(v) for v in t.bbox]
                robots_w.writerow(
                    [frame_idx, round(time_sec, 3), alliance, rid, x, y, w, h, int(t.visible)]
                )
        for bid, t in ball_tracks.items():
            x, y, w, h = [int(v) for v in t.bbox]
            owner = engine.ball_possessor.get(bid)
            owner_str = f"{owner[0]}{owner[1]}" if owner else ""
            balls_w.writerow(
                [frame_idx, round(time_sec, 3), bid, x, y, w, h, int(t.visible), owner_str]
            )

        # --- draw ---
        overlay.draw_zones(frame, zones)
        overlay.draw_possession(frame, ball_tracks, robots_by_alliance, engine)
        overlay.draw_robots(frame, red_tracks, "red", engine)
        overlay.draw_robots(frame, blue_tracks, "blue", engine)
        overlay.draw_balls(frame, ball_tracks, engine)
        overlay.draw_scoreboard(frame, engine, frame_idx, time_sec)
        if flash is not None:
            event, remaining = flash
            overlay.draw_score_flash(frame, event, alpha=min(1.0, remaining / flash_frames))
            flash = (event, remaining - 1) if remaining > 1 else None

        writer.write(frame)
        if show:
            cv2.imshow("FRC Vision", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        frame_idx += 1

    cap.release()
    writer.release()
    robots_csv.close()
    balls_csv.close()
    if show:
        cv2.destroyAllWindows()

    # --- scoring events CSV + summary ---
    with open(events_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame",
                "time_sec",
                "ball_id",
                "zone",
                "credited_alliance",
                "credited_robot",
                "x",
                "y",
            ]
        )
        for e in engine.events:
            w.writerow(
                [
                    e["frame"],
                    e["time_sec"],
                    e["ball_id"],
                    e["zone"],
                    e["credited_alliance"],
                    e["credited_robot"],
                    e["x"],
                    e["y"],
                ]
            )

    summary = build_summary(engine, frame_idx - start_frame, fps)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary, outdir)
    return summary


def build_summary(engine, frames, fps):
    per_robot = [
        {"alliance": a, "robot": rid, "balls_scored": n}
        for (a, rid), n in sorted(engine.robot_scores.items(), key=lambda kv: -kv[1])
    ]
    return {
        "frames_processed": frames,
        "fps": round(fps, 2),
        "duration_sec": round(frames / fps, 2) if fps else None,
        "total_balls_scored": len(engine.events),
        "red_alliance_balls": engine.alliance_total("red"),
        "blue_alliance_balls": engine.alliance_total("blue"),
        "unattributed_balls": engine.unattributed,
        "per_robot": per_robot,
        "events": engine.events,
    }


def print_summary(summary, outdir):
    print("\n" + "=" * 46)
    print("  FRC VISION -- SCORING SUMMARY")
    print("=" * 46)
    print(f"  frames processed : {summary['frames_processed']} ({summary['duration_sec']}s)")
    print(
        f"  balls scored     : {summary['total_balls_scored']}"
        f"   (RED {summary['red_alliance_balls']} / BLUE {summary['blue_alliance_balls']}"
        f" / ? {summary['unattributed_balls']})"
    )
    print("  per robot:")
    if summary["per_robot"]:
        for r in summary["per_robot"]:
            tag = "R" if r["alliance"] == "red" else "B"
            print(f"    {tag}{r['robot']:<3} scored {r['balls_scored']} ball(s)")
    else:
        print("    (none attributed)")
    print("=" * 46)
    print(f"  outputs written to: {outdir}/")
    print("    annotated.mp4, robot_positions.csv, ball_positions.csv,")
    print("    scoring_events.csv, summary.json\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Track FRC robots and the balls they score.")
    p.add_argument("--input", required=True, help="Input match video")
    p.add_argument("--outdir", default="out", help="Directory for all outputs")
    p.add_argument("--zones", help="JSON file of goal zones")
    p.add_argument(
        "--pick-zones",
        action="store_true",
        help="Interactively draw goal zones on the first frame, then exit (or run if --zones-out unused)",
    )
    p.add_argument("--zones-out", help="Where to save picked zones (implies --pick-zones)")
    p.add_argument(
        "--ball-color",
        default=config.DEFAULT_BALL_COLOR,
        choices=sorted(config.BALL_COLOR_PRESETS.keys()),
        help="Game-piece color preset",
    )
    p.add_argument("--show", action="store_true", help="Live preview window while processing")
    p.add_argument(
        "--no-robots",
        action="store_true",
        help="Skip robot detection entirely (ball tracking only)",
    )
    p.add_argument("--start-frame", type=int, default=0, help="Frame to start processing at")
    p.add_argument(
        "--max-frames", type=int, help="Process at most this many frames (quick preview)"
    )
    # tuning overrides
    p.add_argument("--robot-min-area", type=int)
    p.add_argument("--ball-min-area", type=int)
    p.add_argument("--ball-max-area", type=int)
    p.add_argument("--ball-min-circularity", type=float)
    p.add_argument("--possession-radius", type=float)
    args = p.parse_args()

    params = Params()
    for name in (
        "robot_min_area",
        "ball_min_area",
        "ball_max_area",
        "ball_min_circularity",
        "possession_radius",
    ):
        val = getattr(args, name)
        if val is not None:
            setattr(params, name, val)

    ball_ranges = config.BALL_COLOR_PRESETS[args.ball_color]

    # Zone resolution: pick > file > none. No zones just means "no scoring
    # detection" -- detection + tracking still run, which is fine for a preview.
    zones = []
    if args.pick_zones or args.zones_out:
        zones = pick_zones_interactive(args.input)
        out = args.zones_out or "zones.json"
        save_zones(zones, out)
        if args.zones_out and not args.zones:
            # picking-only run: save and exit so the user can review before processing
            print("Zones saved. Re-run with --zones", out, "to process the video.")
            return
    elif args.zones:
        zones = load_zones(args.zones)
    else:
        print("No goal zones given -> running detection + tracking only (no scoring).")

    process_video(
        args.input,
        args.outdir,
        zones,
        ball_ranges,
        params,
        show=args.show,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        detect_robots=not args.no_robots,
    )


if __name__ == "__main__":
    main()
