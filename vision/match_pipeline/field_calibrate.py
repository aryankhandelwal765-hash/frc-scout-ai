"""
One-time field calibration (fixed division camera -> reused for all 140 matches).

Captures, in the detector's CROPPED pixel space:
  - the 6 driver-station start anchors (blue1..blue3, red1..red3) -- where each
    station's robot lines up at the start of auto. These let us map a detected
    robot's start position -> station -> TBA team number (classification).
  - (optional) a pixel->field homography from 4 known field points, for metric
    paths instead of warped-pixel paths.

Saves calib.json + reference_frame.png in match_pipeline/. Because the Hopper camera is
fixed, you calibrate ONCE and every match reuses it.

INTERACTIVE (needs a display window -- run from a Terminal, or PyCharm's Run works):
  python field_calibrate.py --input videos/2026hop_qm1.mp4 --at 6
    -> grabs the frame at t=6s, opens it, click the 6 anchors in the prompted order.

If you can't click (headless), set them from the CLI instead:
  python field_calibrate.py --input videos/2026hop_qm1.mp4 --at 6 \
      --stations blue1=210,300 blue2=250,330 blue3=300,300 \
                 red1=1090,300 red2=1050,330 red3=1000,300
"""

import argparse
import json
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
from boxes import CROP   # shared with trajectories.py and the detector
STATION_ORDER = ["blue1", "blue2", "blue3", "red1", "red2", "red3"]
CALIB = os.path.join(HERE, "calib.json")
REF_IMG = os.path.join(HERE, "reference_frame.png")

# ===========================================================================
# EDIT ME (PyCharm: click Run — no terminal args needed). CLI args override.
# Set AT to a second where the robots are LINED UP at auto start, then click the
# 6 anchors (blue1,blue2,blue3,red1,red2,red3) in that order in the window.
# ===========================================================================
INPUT = os.path.join(HERE, "videos", "2026hop_qm1.mp4")  # video to grab a frame from
AT = 6.0               # seconds into the video for the reference frame
FRAME = ""             # or use an existing cropped image instead of a video
HOMOGRAPHY = False     # True = also click 4 field points + type real coords
STATIONS = []          # non-interactive: ["blue1=x,y", ...]; [] = click them
CLICK_ZONES = False    # True = after anchors, DRAG the blue+red scoring zones
ZONES = []             # non-interactive: ["blue=x,y,w,h", "red=x,y,w,h"]
# ===========================================================================


def grab_frame(input_path, at_sec):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(at_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Could not read frame at {at_sec}s from {input_path}")
    x1, y1, x2, y2 = CROP
    return frame[y1:y2, x1:x2].copy()


def click_points(img, labels, title):
    """Click one point per label, in order. Returns {label: [x, y]}."""
    pts = {}
    disp = img.copy()

    def redraw():
        d = img.copy()
        for i, lb in enumerate(labels):
            if lb in pts:
                x, y = pts[lb]
                cv2.circle(d, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(d, lb, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        nxt = next((lb for lb in labels if lb not in pts), None)
        msg = f"click: {nxt}" if nxt else "all set - press ENTER"
        cv2.rectangle(d, (0, 0), (d.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(d, f"{title}  |  {msg}  |  u=undo ENTER=done ESC=cancel",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return d

    def on_mouse(event, x, y, flags, param):
        nonlocal disp
        if event == cv2.EVENT_LBUTTONDOWN:
            nxt = next((lb for lb in labels if lb not in pts), None)
            if nxt:
                pts[nxt] = [int(x), int(y)]
                disp = redraw()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    disp = redraw()
    while True:
        cv2.imshow(title, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):            # ENTER
            if len(pts) == len(labels):
                break
        elif k == ord("u"):          # undo last
            for lb in reversed(labels):
                if lb in pts:
                    del pts[lb]
                    break
            disp = redraw()
        elif k == 27:                # ESC
            cv2.destroyAllWindows()
            raise SystemExit("Calibration cancelled.")
    cv2.destroyWindow(title)
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=INPUT, help="video to grab a reference frame from")
    p.add_argument("--frame", default=(FRAME or None),
                   help="use this image file as the reference frame (already cropped)")
    p.add_argument("--at", type=float, default=AT, help="seconds into the video to grab")
    p.add_argument("--stations", nargs="*", default=STATIONS, metavar="name=x,y",
                   help="set anchors non-interactively instead of clicking")
    p.add_argument("--homography", action="store_true", default=HOMOGRAPHY,
                   help="also click 4 field points + type their real coords")
    p.add_argument("--click-zones", action="store_true", default=CLICK_ZONES,
                   help="drag the blue+red scoring zones after the anchors")
    p.add_argument("--zones", nargs="*", default=ZONES, metavar="alliance=x,y,w,h",
                   help="set scoring zones non-interactively")
    args = p.parse_args()

    if args.frame:
        img = cv2.imread(args.frame)
        if img is None:
            raise SystemExit(f"Could not read {args.frame}")
    elif args.input:
        img = grab_frame(args.input, args.at)
    else:
        raise SystemExit("Give --input VIDEO or --frame IMAGE.")
    cv2.imwrite(REF_IMG, img)

    if args.stations:
        stations = {}
        for tok in args.stations:
            name, xy = tok.split("=")
            x, y = xy.split(",")
            stations[name] = [int(x), int(y)]
        missing = [s for s in STATION_ORDER if s not in stations]
        if missing:
            raise SystemExit(f"Missing stations: {missing}")
    else:
        print("Click the 6 start anchors in order:", ", ".join(STATION_ORDER))
        stations = click_points(img, STATION_ORDER, "Station start anchors")

    homography = None
    field_pts = None
    if args.homography:
        labels = ["p1", "p2", "p3", "p4"]
        px = click_points(img, labels, "Homography: click 4 known field points")
        field_pts = {}
        print("Enter each clicked point's REAL field coord (e.g. feet), 'x y':")
        import numpy as np
        src, dst = [], []
        for lb in labels:
            fx, fy = input(f"  {lb} field x y: ").split()
            field_pts[lb] = [float(fx), float(fy)]
            src.append(px[lb]); dst.append([float(fx), float(fy)])
        H, _ = cv2.findHomography(np.array(src, float), np.array(dst, float))
        homography = H.tolist()

    # scoring zones (where balls actually get shot in) -- used to ignore floor-ball
    # churn and only count ball activity in the hub areas.
    scoring_zones = []
    if args.zones:
        for tok in args.zones:                       # "blue=x1,y1,x2,y2,x3,y3,x4,y4"
            al, nums = tok.split("=")
            v = [int(n) for n in nums.split(",")]
            scoring_zones.append({"alliance": al,
                                  "poly": [[v[i], v[i + 1]] for i in range(0, len(v), 2)]})
    elif args.click_zones:                           # fisheye: click 4 corners
        for al in ("blue", "red"):
            print(f"Click the 4 corners of the {al.upper()} scoring zone (clockwise).")
            px = click_points(img, ["c1", "c2", "c3", "c4"], f"{al} scoring zone")
            scoring_zones.append({"alliance": al,
                                  "poly": [px["c1"], px["c2"], px["c3"], px["c4"]]})

    calib = {"crop": CROP, "reference_frame": REF_IMG, "stations": stations,
             "homography": homography, "field_points": field_pts,
             "scoring_zones": scoring_zones}
    with open(CALIB, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nSaved {CALIB}")
    print("stations:", stations)
    print("scoring_zones:", scoring_zones)
    if homography:
        print("homography captured (pixel -> field).")


if __name__ == "__main__":
    main()
