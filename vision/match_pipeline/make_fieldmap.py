"""
One-time: fit the camera->top-down field map by placing 9 known field landmarks.

The camera is fixed for the whole event, so this is done ONCE and every match reuses
it. The 9 points are a 3x3 grid on the field FLOOR:

  1 top-left corner   2 top-edge middle    3 top-right corner
  4 left-edge middle  5 field CENTER       6 right-edge middle
  7 bottom-left cnr   8 bottom-edge middle 9 bottom-right corner

The edge/centre points exist because the overhead shot is barrel-distorted: a
4-corner homography only flattens a pinhole view, so a 2nd-order polynomial is
fitted instead and the middles are where the bow is largest.

You do NOT click cold. The 9 points start pre-placed -- carried over from the
previous calibration and rescaled to this resolution if one exists, otherwise laid
out on a default grid -- so the job is dragging each one onto its landmark.

Background is the MEDIAN frame (the empty field), so no robots or balls sit on top
of the corners you are trying to hit.

PyCharm: set MATCH below, click Run.
  drag   move the nearest point        r  reset to the seeded layout
  ENTER  fit + save into calib.json    ESC  cancel
"""

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
import topdown as td

VIDEOS = os.path.join(HERE, "videos")
CALIB = os.path.join(HERE, "calib.json")
REF_IMG = os.path.join(HERE, "reference_frame_1080.png")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
MATCH = "2026hop_qm27"   # which match's field to calibrate on (camera is fixed)
DEGREE = 2               # 2 = quadratic, right for barrel distortion. Needs >= 6 pts.
BLUE_LEFT = True         # which alliance is on the LEFT in the broadcast.
                         # qm100/qm27 auto views show blue left -- change if yours differ.
FORCE_REFRESH = False    # True = rebuild the reference frame even if cached
# ===========================================================================

ORDER = ["1 top-LEFT corner", "2 top-edge MIDDLE", "3 top-RIGHT corner",
         "4 left-edge MIDDLE", "5 field CENTER", "6 right-edge MIDDLE",
         "7 bottom-LEFT corner", "8 bottom-edge MIDDLE", "9 bottom-RIGHT corner"]


def reference_frame(match, force=False):
    """Empty-field background at the CURRENT resolution, cropped with field_crop()."""
    import trajectories as tj
    vid = os.path.join(VIDEOS, f"{match}.mp4")
    if not os.path.exists(vid):
        raise SystemExit(f"No video: {vid}")
    cap = cv2.VideoCapture(vid)
    ok, f0 = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {vid}")
    crop = tj.field_crop(f0.shape)
    x1, y1, x2, y2 = crop
    want = (y2 - y1, x2 - x1)

    if not force and os.path.exists(REF_IMG):
        img = cv2.imread(REF_IMG)
        if img is not None and img.shape[:2] == want:
            return img, crop
        print("  cached reference frame is the wrong size for this resolution "
              "-- rebuilding")
    print("  building empty-field median frame ...")
    med = tj.median_frame(vid, crop)
    img = med if med is not None else f0[y1:y2, x1:x2].copy()
    cv2.imwrite(REF_IMG, img)
    return img, crop


def seed_points(shape):
    """Starting positions: the previous calibration rescaled to this resolution if
    there is one, else an inset 3x3 grid."""
    h, w = shape[:2]
    old = None
    if os.path.exists(CALIB):
        fm = (json.load(open(CALIB)) or {}).get("fieldmap") or {}
        old = fm.get("src_pts")
    if old and len(old) == 9:
        ox = max(p[0] for p in old) or 1
        oy = max(p[1] for p in old) or 1
        # old points were fitted in a smaller crop; rescale by the extent they spanned
        sx, sy = (w * 0.98) / ox, (h * 0.98) / oy
        pts = [(int(px * sx), int(py * sy)) for px, py in old]
        print(f"  seeded from the previous calibration, rescaled x{sx:.2f}/{sy:.2f}"
              "  -- VERIFY every point, the old one was fitted at another resolution")
        return pts
    xs = [int(w * f) for f in (0.06, 0.5, 0.94)]
    ys = [int(h * f) for f in (0.12, 0.5, 0.92)]
    return [(x, y) for y in ys for x in xs]


def place9(img, pts):
    pts = list(pts)
    drag = {"i": None}

    def redraw():
        d = img.copy()
        # the implied grid, so a badly-placed point is obvious at a glance
        for a, b in ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
                     (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)):
            cv2.line(d, pts[a], pts[b], (0, 200, 0), 1, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            col = (0, 255, 255) if drag["i"] == i else (0, 255, 0)
            cv2.drawMarker(d, (x, y), (0, 0, 0), cv2.MARKER_CROSS, 22, 4)
            cv2.drawMarker(d, (x, y), col, cv2.MARKER_CROSS, 20, 2)
            cv2.putText(d, str(i + 1), (x + 9, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4)
            cv2.putText(d, str(i + 1), (x + 9, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, col, 2)
        cv2.rectangle(d, (0, 0), (d.shape[1], 30), (0, 0, 0), -1)
        msg = ("drag points onto: corners / edge middles / centre   "
               "[r=reset  ENTER=save  ESC=cancel]")
        cv2.putText(d, msg, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        return d

    def on_mouse(e, x, y, flags, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            dist, i = min(((x - px) ** 2 + (y - py) ** 2, i)
                          for i, (px, py) in enumerate(pts))
            drag["i"] = i if dist < 60 ** 2 else None
        elif e == cv2.EVENT_MOUSEMOVE and drag["i"] is not None:
            pts[drag["i"]] = (int(x), int(y))
        elif e == cv2.EVENT_LBUTTONUP:
            drag["i"] = None

    win = "field grid -- drag the 9 landmarks"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1700, img.shape[1]), min(900, img.shape[0] + 40))
    cv2.setMouseCallback(win, on_mouse)
    seeded = list(pts)
    while True:
        cv2.imshow(win, redraw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):
            break
        if k == ord("r"):
            pts[:] = list(seeded)
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("cancelled -- nothing saved")
    cv2.destroyAllWindows()
    return pts


def main():
    img, crop = reference_frame(MATCH, FORCE_REFRESH)
    print(f"  reference frame {img.shape[1]}x{img.shape[0]}  (crop {crop})")
    pts = place9(img, seed_points(img.shape))
    td.save_fieldmap(pts, DEGREE)
    print(f"saved fieldmap (degree {DEGREE}) into calib.json:")
    for name, p in zip(ORDER, pts):
        print(f"   {name:24s} {p}")

    # sanity preview: push a dense camera grid through the fit. A good map lands as
    # an even lattice inside the field; bowing means re-place or raise DEGREE.
    fm = td.load_mapper()
    canvas = td.draw_field(blue_left=BLUE_LEFT)
    step = 20
    gy, gx = np.mgrid[0:img.shape[0]:step, 0:img.shape[1]:step]
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    inside = 0
    for (X, Y) in fm(grid):
        if 0 <= X < td.CW and 0 <= Y < td.CH:
            cv2.circle(canvas, (int(X), int(Y)), 1, (0, 0, 0), -1)
            inside += 1
    out = os.path.join(HERE, "fieldmap_preview.png")
    cv2.imwrite(out, canvas)
    print(f"wrote {out}  ({inside}/{len(grid)} grid points landed on the field)")
    print("Check it: the dots should form an even rectangular lattice. Bowed edges "
          "= re-place the edge middles or raise DEGREE.")


if __name__ == "__main__":
    main()
