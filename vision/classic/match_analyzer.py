"""
FRC Match Analyzer: robot tracking + game piece tracking + score attribution.

Extends robot_tracker.py's bumper-color detection with:
  - Game piece (ball) detection via HSV + circularity filtering
  - Goal zone polygons (from calibrate.py's config.json)
  - Attribution: when a piece enters a goal zone, credit whichever tracked
    robot was closest to that piece's earliest tracked position

Usage:
    python match_analyzer.py --input clip.mp4 --config config.json \
        --output annotated.mp4 --events scoring_events.csv
"""

import argparse
import csv
import json
from collections import defaultdict

import cv2
import numpy as np
from robot_tracker import BLUE_RANGE, MIN_CONTOUR_AREA, RED_RANGES, find_blobs, get_color_mask


class HistoryTracker:
    """Like robot_tracker.SimpleTracker, but also keeps a full position history
    per track ID so we can look up "where was this track N frames ago"."""

    def __init__(self, max_distance=80):
        self.max_distance = max_distance
        self.next_id = 0
        self.tracks = {}  # id -> last centroid
        self.history = defaultdict(list)  # id -> [(frame_idx, cx, cy), ...]

    def update(self, detections, frame_idx):
        assigned = {}
        used_ids = set()
        for det in detections:
            cx, cy = det["centroid"]
            best_id, best_dist = None, self.max_distance
            for tid, (px, py) in self.tracks.items():
                if tid in used_ids:
                    continue
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = tid, dist
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used_ids.add(best_id)
            self.tracks[best_id] = (cx, cy)
            self.history[best_id].append((frame_idx, cx, cy))
            assigned[best_id] = det
        self.tracks = {tid: c for tid, c in self.tracks.items() if tid in used_ids}
        return assigned

    def position_near_frame(self, track_id, frame_idx, window=15):
        """Closest recorded position for this track within `window` frames of frame_idx."""
        best, best_dist = None, window + 1
        for f, cx, cy in self.history.get(track_id, []):
            d = abs(f - frame_idx)
            if d < best_dist:
                best, best_dist = (cx, cy), d
        return best


def find_ball_blobs(hsv, lower, upper, min_area, max_area, min_circularity):
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity < min_circularity:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        blobs.append({"bbox": (x, y, w, h), "centroid": (cx, cy), "area": area})
    return blobs


def point_in_zone(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0


def attribute_score(ball_history, robot_trackers):
    """Given a ball's full position history, find which robot (across both
    alliance trackers) was closest to the ball's earliest recorded position."""
    origin_frame, ox, oy = ball_history[0]
    best_label, best_dist = None, float("inf")
    for alliance, tracker in robot_trackers.items():
        for tid in tracker.history:
            pos = tracker.position_near_frame(tid, origin_frame)
            if pos is None:
                continue
            dist = ((pos[0] - ox) ** 2 + (pos[1] - oy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_label = (alliance, tid)
    return best_label


def process_video(input_path, config_path, output_path, events_path, min_robot_area):
    with open(config_path) as f:
        config = json.load(f)

    ball_lower = config["ball_hsv"]["lower"]
    ball_upper = config["ball_hsv"]["upper"]
    ball_min_area = config.get("ball_min_area", 50)
    ball_max_area = config.get("ball_max_area", 3000)
    ball_min_circularity = config.get("ball_min_circularity", 0.55)
    goal_zones = config["goal_zones"]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise OSError(f"Could not open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    robot_trackers = {"red": HistoryTracker(), "blue": HistoryTracker()}
    ball_tracker = HistoryTracker(max_distance=100)
    scored_ball_ids = set()
    score_summary = defaultdict(int)  # (alliance, robot_id) -> count

    events_file = open(events_path, "w", newline="")
    events_writer = csv.writer(events_file)
    events_writer.writerow(
        ["frame", "time_sec", "zone", "attributed_alliance", "attributed_robot_id"]
    )

    kernel = np.ones((5, 5), np.uint8)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- robots ---
        red_mask = cv2.morphologyEx(get_color_mask(hsv, RED_RANGES), cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(get_color_mask(hsv, [BLUE_RANGE]), cv2.MORPH_OPEN, kernel)
        red_blobs = find_blobs(red_mask, min_robot_area)
        blue_blobs = find_blobs(blue_mask, min_robot_area)
        red_tracks = robot_trackers["red"].update(red_blobs, frame_idx)
        blue_tracks = robot_trackers["blue"].update(blue_blobs, frame_idx)

        for tid, det in red_tracks.items():
            x, y, w, h = det["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"R{tid}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for tid, det in blue_tracks.items():
            x, y, w, h = det["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"B{tid}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --- game pieces ---
        ball_blobs = find_ball_blobs(
            hsv, ball_lower, ball_upper, ball_min_area, ball_max_area, ball_min_circularity
        )
        ball_tracks = ball_tracker.update(ball_blobs, frame_idx)

        for tid, det in ball_tracks.items():
            cx, cy = det["centroid"]
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 255, 255), 2)

            if tid in scored_ball_ids:
                continue

            for zone_name, polygon in goal_zones.items():
                if point_in_zone((cx, cy), polygon):
                    attribution = attribute_score(ball_tracker.history[tid], robot_trackers)
                    scored_ball_ids.add(tid)
                    time_sec = frame_idx / fps
                    if attribution:
                        alliance, robot_id = attribution
                        score_summary[(alliance, robot_id)] += 1
                        events_writer.writerow(
                            [frame_idx, round(time_sec, 3), zone_name, alliance, robot_id]
                        )
                        cv2.putText(
                            frame,
                            f"SCORE: {alliance.upper()}{robot_id} -> {zone_name}",
                            (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        events_writer.writerow(
                            [frame_idx, round(time_sec, 3), zone_name, "unknown", ""]
                        )
                    break

        # draw goal zones for reference
        for zone_name, polygon in goal_zones.items():
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    events_file.close()

    print(f"Processed {frame_idx} frames -> {output_path}, {events_path}")
    print("Score summary:")
    for (alliance, robot_id), count in sorted(score_summary.items()):
        print(f"  {alliance.upper()}{robot_id}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track robots + game pieces and attribute scores in FRC match video."
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--config", required=True, help="Path to config.json from calibrate.py")
    parser.add_argument("--output", default="annotated.mp4", help="Path to annotated output video")
    parser.add_argument(
        "--events", default="scoring_events.csv", help="Path to output CSV of scoring events"
    )
    parser.add_argument("--min-robot-area", type=int, default=MIN_CONTOUR_AREA)
    args = parser.parse_args()

    process_video(args.input, args.config, args.output, args.events, args.min_robot_area)
