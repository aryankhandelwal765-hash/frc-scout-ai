"""
One-time: click the 4 corners of the FIELD FLOOR in the broadcast to build the
camera->top-down homography (fixed camera -> reused for all matches).

Click the corners of the black field-boundary rectangle in this order:
  1 top-left  2 top-right  3 bottom-right  4 bottom-left
(as they appear on screen). Saved into calib.json under "topdown".

PyCharm: click Run. Uses reference_frame.png if present, else grabs from INPUT@AT.
u = undo, ENTER = done, ESC = cancel.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import topdown as td

CROP = (0, 100, 1280, 470)
REF_IMG = os.path.join(HERE, "reference_frame.png")

# ===========================================================================
INPUT = os.path.join(HERE, "videos", "2026hop_qm27.mp4")
AT = 8.0
CORNERS = []    # non-interactive: ["x,y","x,y","x,y","x,y"] TL,TR,BR,BL; [] = click
# ===========================================================================
LABELS = ["1 top-left", "2 top-right", "3 bottom-right", "4 bottom-left"]


def _frame():
    if os.path.exists(REF_IMG):
        img = cv2.imread(REF_IMG)
        if img is not None:
            return img
    cap = cv2.VideoCapture(INPUT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(AT * fps))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {INPUT}")
    x1, y1, x2, y2 = CROP
    img = f[y1:y2, x1:x2].copy()
    cv2.imwrite(REF_IMG, img)
    return img


def click4(img):
    pts = []

    def redraw():
        d = img.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(d, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(d, str(i + 1), (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(pts) >= 2:
            cv2.polylines(d, [np.array(pts, np.int32)], len(pts) == 4, (0, 255, 0), 2)
        nxt = LABELS[len(pts)] if len(pts) < 4 else "ENTER to save"
        cv2.rectangle(d, (0, 0), (d.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(d, f"click field corner: {nxt}   (u=undo ENTER=done ESC=cancel)",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return d

    def on_mouse(e, x, y, flags, p):
        if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((int(x), int(y)))

    cv2.namedWindow("field corners")
    cv2.setMouseCallback("field corners", on_mouse)
    while True:
        cv2.imshow("field corners", redraw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) == 4:
            break
        if k == ord("u") and pts:
            pts.pop()
        if k == 27:
            raise SystemExit("cancelled")
    cv2.destroyAllWindows()
    return pts


def main():
    img = _frame()
    if CORNERS:
        pts = [tuple(int(v) for v in c.split(",")) for c in CORNERS]
    else:
        pts = click4(img)
    td.save_homography(pts)
    print("saved homography; corners:", pts)
    # sanity preview: warp the reference frame to top-down
    H = td.load_homography()
    warp = cv2.warpPerspective(img, H, (td.CW, td.CH))
    cv2.imwrite(os.path.join(HERE, "homography_preview.png"), warp)
    print("wrote homography_preview.png (the field warped flat -- check it looks rectangular)")


if __name__ == "__main__":
    main()
