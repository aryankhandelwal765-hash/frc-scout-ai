"""
Calibration tool for match_analyzer.py

Run this LOCALLY on your Mac (needs a display - it opens an OpenCV window,
which won't work in a headless environment).

Usage:
    python calibrate.py --input clip.mp4 --output config.json

Controls:
    - Starts in BALL mode: click on 3-5 visible game pieces in the frame.
    - Press 'z' to switch to ZONE mode: click points to outline a goal zone
      polygon (e.g. the hopper opening), then press 'n' to name and save it,
      and start outlining the next zone.
    - Press 's' to save config.json and quit.
    - Press 'q' to quit without saving.
    - Press 'r' to reset everything and start over.
"""

import argparse
import json

import cv2
import numpy as np


def sample_hsv_around(hsv_frame, x, y, radius=6):
    h, w = hsv_frame.shape[:2]
    y0, y1 = max(0, y - radius), min(h, y + radius)
    x0, x1 = max(0, x - radius), min(w, x + radius)
    patch = hsv_frame[y0:y1, x0:x1].reshape(-1, 3)
    return patch.mean(axis=0)  # (H, S, V) mean


def compute_hsv_range(samples, h_tol=12, s_tol=60, v_tol=60):
    samples = np.array(samples)
    mean = samples.mean(axis=0)
    lower = np.clip([mean[0] - h_tol, mean[1] - s_tol, mean[2] - v_tol], 0, 255)
    upper = np.clip([mean[0] + h_tol, mean[1] + s_tol, mean[2] + v_tol], 0, 255)
    return lower.astype(int).tolist(), upper.astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate ball color + goal zones for match_analyzer.py"
    )
    parser.add_argument("--input", required=True, help="Path to a representative match video")
    parser.add_argument(
        "--output", default="config.json", help="Where to save the calibration config"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=60,
        help="Which frame to calibrate on (default: 60, to skip intros)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    if not ret:
        raise OSError("Could not read a frame from the video at that position.")
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    state = {
        "mode": "ball",
        "ball_samples": [],
        "zones": {},
        "current_zone_points": [],
    }

    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["mode"] == "ball":
            hsv_val = sample_hsv_around(hsv_frame, x, y)
            state["ball_samples"].append(hsv_val)
            print(
                f"Sampled ball HSV ~ {hsv_val.round(1)} ({len(state['ball_samples'])} samples so far)"
            )
        elif state["mode"] == "zone":
            state["current_zone_points"].append([x, y])

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", mouse_cb)

    print(__doc__)

    while True:
        disp = frame.copy()

        for name, poly in state["zones"].items():
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(disp, [pts], True, (0, 255, 255), 2)
            cv2.putText(disp, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if state["mode"] == "zone" and state["current_zone_points"]:
            pts = np.array(state["current_zone_points"], dtype=np.int32)
            cv2.polylines(disp, [pts], False, (0, 255, 0), 2)
            for p in state["current_zone_points"]:
                cv2.circle(disp, tuple(p), 3, (0, 255, 0), -1)

        for s_hsv in state["ball_samples"]:
            pass  # samples don't have pixel coords stored; visual feedback is via the printed HSV

        mode_text = f"MODE: {state['mode'].upper()}  (z=zone mode, n=name+save zone, s=save+quit, q=quit, r=reset)"
        cv2.putText(disp, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("calibrate", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("z"):
            state["mode"] = "zone"
            print("Switched to ZONE mode. Click polygon points, then press 'n' to name+save it.")
        elif key == ord("n"):
            if state["mode"] == "zone" and len(state["current_zone_points"]) >= 3:
                name = input("Name this goal zone (e.g. red_hopper, blue_hopper): ").strip()
                if name:
                    state["zones"][name] = state["current_zone_points"]
                    print(f"Saved zone '{name}' with {len(state['current_zone_points'])} points.")
                state["current_zone_points"] = []
                state["mode"] = "ball"
                print("Back to BALL mode (press 'z' again for another zone).")
            else:
                print("Need at least 3 points to define a zone polygon.")
        elif key == ord("r"):
            state = {"mode": "ball", "ball_samples": [], "zones": {}, "current_zone_points": []}
            print("Reset.")
        elif key == ord("s"):
            if len(state["ball_samples"]) < 2:
                print("Sample at least 2 ball colors before saving (press 'q' to abort instead).")
                continue
            if not state["zones"]:
                print(
                    "Define at least 1 goal zone before saving (press 'z' then click points, 'n' to name)."
                )
                continue
            lower, upper = compute_hsv_range(state["ball_samples"])
            config = {
                "ball_hsv": {"lower": lower, "upper": upper},
                "ball_min_area": 50,
                "ball_max_area": 3000,
                "ball_min_circularity": 0.55,
                "goal_zones": state["zones"],
            }
            with open(args.output, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Saved config to {args.output}")
            break
        elif key == ord("q"):
            print("Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
