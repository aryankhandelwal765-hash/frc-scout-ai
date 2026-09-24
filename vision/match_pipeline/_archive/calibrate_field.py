"""
Calibrate camera -> field using REAL structures at manual-known positions.

Why this replaces autofit_fieldmap.py. That one inferred the field outline from the
robot point cloud, but robots do not drive to the carpet edges -- least of all the
far sideline -- so the fitted field was inset, and inset WORST in the depth
direction. Measured: the autofit spanned 95% of the image horizontally but only 77%
vertically, which squeezes every mapped position toward mid-field in depth. That is
the depth error visible in the match renders.

Fix: click things whose real position is known, not things robots happen to reach.

FIELD COORDINATES (inches), from the 2026 REBUILT manual:
    X runs along the FIELD LENGTH:  0 = BLUE ALLIANCE WALL, 651.2 = RED ALLIANCE WALL
    Y runs across the FIELD WIDTH:  0 = one guardrail,       317.7 = the other

Landmarks and where the numbers come from:
    ROBOT STARTING LINE   X = 158.6 (blue) / 492.6 (red)
        manual 5.3: the ALLIANCE ZONE is 158.6in deep and "is bounded by and
        includes the ROBOT STARTING LINE"
    CENTER LINE           X = 325.6
        manual 5.3: "spans the width of the FIELD" and bisects the NEUTRAL ZONE
    HUB                   47.0in square, near face on the ROBOT STARTING LINE,
                          centred across the width -> Y = 135.35 and 182.35
        manual 5.4: "one of two 47in by 47in" and "centered between two BUMPS
        located 158.6in away from their ALLIANCE WALL"
    BUMP                  73.0in wide, flanking the HUB
        -> outer edges at Y = 62.35 and 255.35   (manual 5.5)

ASSUMPTION worth knowing: that the HUB's NEAR FACE (not its centre) sits at 158.6in.
The manual says the HUB is "located 158.6in away from their ALLIANCE WALL" without
saying to which face. If that is wrong the fit residuals will show it -- this script
prints per-point error in inches so a bad assumption is visible, not silent.

CONTROLS
    click   place the current point        s  skip a point you cannot see clearly
    drag    move an already-placed point   u  undo the last placement
    ENTER   fit and save                   ESC  cancel, save nothing

PyCharm: set MATCH, click Run.
"""

import argparse
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
REF = os.path.join(HERE, "reference_field.png")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
MATCH = "2026hop_qm27"
BLUE_ON_LEFT = True     # is the BLUE alliance wall on the LEFT of the broadcast?
                        # the auto views show blue left for 2026hop.
DEGREE = 2              # polynomial degree; 2 absorbs the barrel distortion
FORCE_REFRESH = False   # rebuild the empty-field background even if cached
# ===========================================================================

FIELD_LEN = 651.2
FIELD_WID = 317.7
START_X = 158.6                     # ROBOT STARTING LINE / HUB near face
CENTER_X = FIELD_LEN / 2.0          # 325.6
HUB_HALF = 47.0 / 2.0               # 23.5
MID_Y = FIELD_WID / 2.0             # 158.85
HUB_Y0, HUB_Y1 = MID_Y - HUB_HALF, MID_Y + HUB_HALF        # 135.35, 182.35
BUMP_Y0, BUMP_Y1 = HUB_Y0 - 73.0, HUB_Y1 + 73.0            # 62.35, 255.35

# (label, X, Y). "TOP"/"BOTTOM" = top/bottom edge of the broadcast frame.
def landmarks(blue_left=True):
    bx = START_X if blue_left else FIELD_LEN - START_X
    rx = FIELD_LEN - START_X if blue_left else START_X
    L = []
    for tag, x in (("BLUE", bx), ("RED", rx)):
        L += [
            (f"{tag} STARTING LINE at the TOP guardrail", x, 0.0),
            (f"{tag} STARTING LINE where the far BUMP's outer edge meets it", x, BUMP_Y0),
            (f"{tag} HUB near-face corner, TOP side", x, HUB_Y0),
            (f"{tag} HUB near-face corner, BOTTOM side", x, HUB_Y1),
            (f"{tag} STARTING LINE where the near BUMP's outer edge meets it", x, BUMP_Y1),
            (f"{tag} STARTING LINE at the BOTTOM guardrail", x, FIELD_WID),
        ]
    bw = 0.0 if blue_left else FIELD_LEN
    rw = FIELD_LEN if blue_left else 0.0
    L += [
        ("CENTER LINE at the TOP guardrail", CENTER_X, 0.0),
        ("CENTER LINE at the BOTTOM guardrail", CENTER_X, FIELD_WID),
        # FIELD CORNERS. The HUBS block the far (top) half at the starting-line X,
        # so without these the fit has almost nothing on the far side and has to
        # extrapolate across it -- the very error this calibration exists to fix.
        # The corners sit at the extremes of X, clear of the hub structures.
        ("FIELD CORNER: TOP guardrail meets the BLUE ALLIANCE WALL", bw, 0.0),
        ("FIELD CORNER: BOTTOM guardrail meets the BLUE ALLIANCE WALL", bw, FIELD_WID),
        ("FIELD CORNER: TOP guardrail meets the RED ALLIANCE WALL", rw, 0.0),
        ("FIELD CORNER: BOTTOM guardrail meets the RED ALLIANCE WALL", rw, FIELD_WID),
    ]
    return L


def reference_frame(match, force=False):
    import trajectories as tj
    vid = os.path.join(VIDEOS, f"{match}.mp4")
    if not os.path.exists(vid):
        raise SystemExit(f"No video: {vid}")
    cap = cv2.VideoCapture(vid)
    ok, f0 = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read {vid}")
    crop = tj.field_crop(f0.shape)
    x1, y1, x2, y2 = crop
    want = (y2 - y1, x2 - x1)
    if not force and os.path.exists(REF):
        img = cv2.imread(REF)
        if img is not None and img.shape[:2] == want:
            return img, crop
    print("  building empty-field median frame ...")
    med = tj.median_frame(vid, crop)
    img = med if med is not None else f0[y1:y2, x1:x2].copy()
    cv2.imwrite(REF, img)
    return img, crop


def start_frame_img(match):
    """The auto-start frame. Robots must have BUMPERS overlapping their ROBOT
    STARTING LINE at MATCH start, so this frame shows exactly where those lines
    are -- far easier to find than hunting for tape on an empty carpet."""
    import trajectories as tj
    vid = os.path.join(VIDEOS, f"{match}.mp4")
    js = os.path.join(HERE, "auto_start.json")
    sf = 0
    if os.path.exists(js):
        d = json.load(open(js)).get(match)
        if d:
            sf = max(0, int(d["start_frame"]))
    cap = cv2.VideoCapture(vid)
    ok, f0 = cap.read()
    if not ok:
        cap.release()
        return None
    x1, y1, x2, y2 = tj.field_crop(f0.shape)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    ok, fr = cap.read()
    cap.release()
    return fr[y1:y2, x1:x2].copy() if ok else None


def place(img, marks, alt=None):
    pts = [None] * len(marks)
    cur = {"i": 0}
    drag = {"i": None}
    bg = {"alt": False}

    def redraw():
        d = (alt if (bg["alt"] and alt is not None) else img).copy()
        for i, p in enumerate(pts):
            if p is None:
                continue
            col = (0, 255, 255) if i == cur["i"] else (0, 255, 0)
            cv2.drawMarker(d, p, (0, 0, 0), cv2.MARKER_CROSS, 24, 5)
            cv2.drawMarker(d, p, col, cv2.MARKER_CROSS, 22, 2)
            cv2.putText(d, str(i + 1), (p[0] + 9, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(d, str(i + 1), (p[0] + 9, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        done = sum(1 for p in pts if p is not None)
        cv2.rectangle(d, (0, 0), (d.shape[1], 56), (0, 0, 0), -1)
        i = cur["i"]
        msg = (f"[{i + 1}/{len(marks)}]  CLICK: {marks[i][0]}"
               if i < len(marks) else "all placed")
        cv2.putText(d, msg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        cv2.putText(d, f"placed {done}  |  click=place  drag=adjust  s=skip  u=undo"
                       f"  b=toggle empty/auto-start view  ENTER=fit+save  ESC=cancel",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return d

    def on_mouse(e, x, y, flags, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            near = [(abs(x - q[0]) ** 2 + abs(y - q[1]) ** 2, i)
                    for i, q in enumerate(pts) if q is not None]
            if near:
                dist, i = min(near)
                if dist < 40 ** 2:
                    drag["i"] = i
                    cur["i"] = i
                    return
            if cur["i"] < len(marks):
                pts[cur["i"]] = (int(x), int(y))
                nxt = cur["i"] + 1
                while nxt < len(marks) and pts[nxt] is not None:
                    nxt += 1
                cur["i"] = nxt
        elif e == cv2.EVENT_MOUSEMOVE and drag["i"] is not None:
            pts[drag["i"]] = (int(x), int(y))
        elif e == cv2.EVENT_LBUTTONUP:
            drag["i"] = None

    win = "field calibration -- click the named landmark"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1800, img.shape[1]), min(950, img.shape[0] + 60))
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, redraw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10):
            break
        if k == ord("s") and cur["i"] < len(marks):
            cur["i"] += 1
        if k == ord("b"):
            bg["alt"] = not bg["alt"]
        if k == ord("u"):
            for i in range(len(pts) - 1, -1, -1):
                if pts[i] is not None:
                    pts[i] = None
                    cur["i"] = i
                    break
        if k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("cancelled -- nothing saved")
    cv2.destroyAllWindows()
    return pts


def fit_and_save(marks, pts, degree=DEGREE):
    have = [(p, m) for p, m in zip(pts, marks) if p is not None]
    if len(have) < 6:
        raise SystemExit(f"only {len(have)} points placed; need at least 6")
    src = np.array([p for p, _ in have], np.float64)
    # field inches -> canonical top-down canvas px
    sx = (td.FX1 - td.FX0) / FIELD_LEN
    sy = (td.FY1 - td.FY0) / FIELD_WID
    dst = np.array([[td.FX0 + m[1] * sx, td.FY0 + m[2] * sy] for _, m in have],
                   np.float64)

    in_per_px = 1.0 / ((sx + sy) / 2.0)

    def errs(pred):
        return np.linalg.norm(pred - dst, axis=1) * in_per_px

    # --- HOMOGRAPHY: the carpet is a plane, so this is the physically correct
    # model, and being a global projective fit it extrapolates sanely into
    # regions with no landmarks.
    H, _ = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), 0)
    h_err = errs(cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2))

    # --- POLYNOMIAL: absorbs barrel distortion, but ONLY where it has data.
    # With landmarks clustered on one side of the field it will swing violently
    # across the unconstrained half, so it is not chosen blindly.
    A = td._design(src, degree)
    cx, *_ = np.linalg.lstsq(A, dst[:, 0], rcond=None)
    cy, *_ = np.linalg.lstsq(A, dst[:, 1], rcond=None)
    poly = {"cx": cx.tolist(), "cy": cy.tolist(), "degree": degree}
    p_err = errs(td.apply_fieldmap(poly, src))

    ys = np.array([m[2] for _, m in have])
    covered = (ys.max() - ys.min()) / FIELD_WID
    n_far = int((ys < FIELD_WID * 0.4).sum())
    print(f"\nfitted on {len(have)} landmarks")
    print(f"  width coverage {covered * 100:.0f}%  |  landmarks on the far half: {n_far}")
    print(f"  homography : mean {h_err.mean():5.1f} in, max {h_err.max():5.1f} in")
    print(f"  polynomial : mean {p_err.mean():5.1f} in, max {p_err.max():5.1f} in "
          f"(degree {degree})")

    # A polynomial always fits the CLICKED points better -- more free parameters.
    # That says nothing about the field where nobody clicked. Only prefer it when
    # the far half is genuinely sampled.
    use_poly = n_far >= 3 and p_err.mean() < h_err.mean() * 0.6
    if use_poly:
        fm = dict(poly)
        chosen, err_in = "polynomial", p_err
    else:
        # express the homography in the same polynomial container the rest of the
        # code already loads, by fitting a degree-1 map to homography output on a
        # dense grid -- exact, since a homography is not affine, so instead store
        # it natively and let load_mapper use it.
        fm = {"H": H.tolist(), "kind": "homography"}
        chosen, err_in = "homography", h_err
        if n_far < 3:
            print(f"  -> using HOMOGRAPHY: only {n_far} landmark(s) on the far half,")
            print("     so a polynomial would be extrapolating blind across it.")
    fm["src_pts"] = src.tolist()
    fm["landmarks"] = [{"label": m[0], "x_in": m[1], "y_in": m[2]}
                       for _, m in have]
    fm["chosen"] = chosen

    print(f"\n  per-point error ({chosen}):")
    for (p, m), e in zip(have, err_in):
        print(f"     {e:6.1f} in   {m[0][:56]}")
    if err_in.max() > 24:
        print("  WARNING: a point is off by more than 2 feet. Either it was placed")
        print("  wrong, or the HUB-near-face assumption is wrong. Check the worst")
        print("  point above before trusting this fit.")

    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    if "fieldmap" in calib and "fieldmap_autofit_backup" not in calib:
        calib["fieldmap_autofit_backup"] = calib["fieldmap"]
    calib["fieldmap"] = fm
    json.dump(calib, open(CALIB, "w"), indent=2)
    print(f"\nsaved -> {CALIB}  (previous autofit kept as fieldmap_autofit_backup)")
    return fm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", default=MATCH)
    p.add_argument("--blue-left", type=int, default=int(BLUE_ON_LEFT))
    a = p.parse_args()

    img, crop = reference_frame(a.match, FORCE_REFRESH)
    print(f"  reference {img.shape[1]}x{img.shape[0]} (crop {crop})")
    marks = landmarks(bool(a.blue_left))
    print(f"  {len(marks)} landmarks to place. Skip any you cannot see clearly.")
    alt = start_frame_img(a.match)
    if alt is not None and alt.shape[:2] == img.shape[:2]:
        print("  press 'b' to toggle between the empty field and the auto-start "
              "frame (robots sit ON their starting lines there)")
    else:
        alt = None
    pts = place(img, marks, alt)
    fit_and_save(marks, pts)

    # how much of the frame does the new fit actually span, vs the autofit's 77%?
    ys = [p[1] for p in pts if p is not None]
    if ys:
        print(f"\n  vertical span of your landmarks: {min(ys)}..{max(ys)} "
              f"= {max(ys) - min(ys)}px of {img.shape[0]}px "
              f"({(max(ys) - min(ys)) / img.shape[0] * 100:.0f}%)")
        print("  the autofit only spanned 77% -- more is better, it means the fit")
        print("  is constrained by real depth rather than extrapolated into it.")
    print("\nNext: re-run topdown_paths.py and render_full_match.py to see the fix.")


if __name__ == "__main__":
    main()
