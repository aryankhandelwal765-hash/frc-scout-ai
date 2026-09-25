"""
Make ONLY the scoring zones as 4-corner QUADRILATERALS (fisheye-friendly).

The overhead shot is barrel-distorted, so a hub isn't an axis-aligned rectangle --
it's a skewed quad. So you click the 4 corners of each alliance's shoot area
instead of dragging a box. Stored as a polygon; merged into calib.json without
touching stations/homography.

PyCharm: click Run. For BLUE then RED, click the 4 corners (clockwise from
top-left), ENTER. u=undo, ESC=skip that zone.
"""

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from boxes import CROP
CALIB = os.path.join(HERE, "calib.json")
REF_IMG = os.path.join(HERE, "reference_frame.png")

# ===========================================================================
# EDIT ME. Click Run. Uses reference_frame.png if it exists, else grabs from INPUT.
# ===========================================================================
INPUT = os.path.join(HERE, "videos", "2026hop_qm1.mp4")
AT = 8.0               # seconds to grab a frame from (only if no reference_frame.png)
# non-interactive fallback: "blue=x1,y1,x2,y2,x3,y3,x4,y4" (4 corners), else [] = click
ZONES = []
# ===========================================================================


def _frame():
    if os.path.exists(REF_IMG):
        img = cv2.imread(REF_IMG)
        if img is not None:
            return img
    cap = cv2.VideoCapture(INPUT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(AT * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {INPUT}")
    x1, y1, x2, y2 = CROP
    img = frame[y1:y2, x1:x2].copy()
    cv2.imwrite(REF_IMG, img)
    return img


def click_quad(img, title, col):
    """Click 4 corners (clockwise from top-left). Returns [[x,y]*4] or None if skipped."""
    pts = []

    def redraw():
        d = img.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(d, (x, y), 5, col, -1)
            cv2.putText(d, str(i + 1), (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if len(pts) >= 2:
            cv2.polylines(d, [np.array(pts, np.int32)], len(pts) == 4, col, 2)
        cv2.rectangle(d, (0, 0), (d.shape[1], 26), (0, 0, 0), -1)
        left = 4 - len(pts)
        msg = f"click corner {len(pts)+1}/4" if left else "4 set - ENTER"
        cv2.putText(d, f"{title} | {msg} | u=undo ENTER=done ESC=skip",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return d

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((int(x), int(y)))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        cv2.imshow(title, redraw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) == 4:
            break
        if k == ord("u") and pts:
            pts.pop()
        if k == 27:
            cv2.destroyWindow(title)
            return None
    cv2.destroyWindow(title)
    return [[x, y] for x, y in pts]


def main():
    img = _frame()
    zones = []
    if ZONES:
        for tok in ZONES:
            al, nums = tok.split("=")
            v = [int(n) for n in nums.split(",")]
            poly = [[v[i], v[i + 1]] for i in range(0, len(v), 2)]
            zones.append({"alliance": al, "poly": poly})
    else:
        for al in ("blue", "red"):
            col = (255, 120, 0) if al == "blue" else (0, 80, 255)
            print(f"Click the 4 corners of the {al.upper()} scoring zone (clockwise).")
            poly = click_quad(img, f"{al} scoring zone", col)
            if poly:
                zones.append({"alliance": al, "poly": poly})

    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {
        "crop": list(CROP), "reference_frame": REF_IMG, "stations": {},
        "homography": None, "field_points": None}
    calib["scoring_zones"] = zones
    with open(CALIB, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nSaved {len(zones)} scoring zone(s) to {CALIB}")


if __name__ == "__main__":
    main()
