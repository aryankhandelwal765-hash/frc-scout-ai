"""
FRC Match Video Robot Tracker
Detects and tracks robots by alliance color (red/blue bumpers) in FRC match footage.

Usage:
    python robot_tracker.py --input match_clip.mp4 --output annotated.mp4 --csv positions.csv
"""

import cv2
import numpy as np
import csv
import argparse

# HSV color ranges for FRC bumpers.
# Red wraps around hue 0/180 in OpenCV's HSV space, so it needs two ranges.
RED_RANGES = [
    ((0, 120, 70), (10, 255, 255)),
    ((170, 120, 70), (180, 255, 255)),
]
BLUE_RANGE = ((100, 120, 70), (130, 255, 255))

MIN_CONTOUR_AREA = 800  # tune based on video resolution / how far the camera is from the field


def get_color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    return mask


def find_blobs(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        blobs.append({"bbox": (x, y, w, h), "centroid": (cx, cy), "area": area})
    return blobs


class SimpleTracker:
    """Nearest-centroid tracker. Assigns a persistent ID to each blob across frames
    so you can follow one robot over time instead of just seeing per-frame boxes."""

    def __init__(self, max_distance=80):
        self.max_distance = max_distance
        self.next_id = 0
        self.tracks = {}  # id -> last centroid

    def update(self, detections):
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
            assigned[best_id] = det
        # drop tracks that weren't matched this frame (robot left frame / occluded)
        self.tracks = {tid: c for tid, c in self.tracks.items() if tid in used_ids}
        return assigned


def process_video(input_path, output_path, csv_path, min_area):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    red_tracker = SimpleTracker()
    blue_tracker = SimpleTracker()

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "time_sec", "alliance", "track_id", "x", "y", "w", "h"])

    kernel = np.ones((5, 5), np.uint8)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = get_color_mask(hsv, RED_RANGES)
        blue_mask = get_color_mask(hsv, [BLUE_RANGE])

        # clean up salt-and-pepper noise from compression artifacts / crowd colors
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        red_blobs = find_blobs(red_mask, min_area)
        blue_blobs = find_blobs(blue_mask, min_area)

        red_tracks = red_tracker.update(red_blobs)
        blue_tracks = blue_tracker.update(blue_blobs)

        time_sec = frame_idx / fps

        for tid, det in red_tracks.items():
            x, y, w, h = det["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"R{tid}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            csv_writer.writerow([frame_idx, round(time_sec, 3), "red", tid, x, y, w, h])

        for tid, det in blue_tracks.items():
            x, y, w, h = det["bbox"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"B{tid}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            csv_writer.writerow([frame_idx, round(time_sec, 3), "blue", tid, x, y, w, h])

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    print(f"Processed {frame_idx} frames -> {output_path}, {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track FRC robots by bumper color in match video.")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="annotated.mp4", help="Path to annotated output video")
    parser.add_argument("--csv", default="positions.csv", help="Path to output CSV of tracked positions")
    parser.add_argument("--min-area", type=int, default=MIN_CONTOUR_AREA,
                         help="Minimum contour area (px) to count as a robot blob")
    args = parser.parse_args()

    process_video(args.input, args.output, args.csv, args.min_area)
