"""
Top-down field model + camera->field homography, for drawing auto paths on a clean
bird's-eye field (like the game manual diagram) instead of the fisheye broadcast.

The floor is a plane, so a 4-point homography maps a robot's ground point (bottom-
centre of its box, in CROP pixels) to a canonical top-down field. Not perfect under
heavy fisheye -- straight lines bow a little near the edges -- but plenty for a
readable auto book. Undistortion (camera calibration) would remove the residual bow.

Canonical field: blue alliance LEFT, red RIGHT (to match the broadcast), so clicking
corners left->right in make_homography lines up with what you see.
"""

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(HERE, "calib.json")

# canonical top-down canvas (pixels). The field is ~2:1, matching the game diagram.
CW, CH = 1640, 860
MARGIN_X, MARGIN_Y = 100, 80
FX0, FY0 = MARGIN_X, MARGIN_Y
FX1, FY1 = CW - MARGIN_X, CH - MARGIN_Y                         # field rectangle
MARGIN = MARGIN_X                                               # back-compat
FIELD_CORNERS = np.array([[FX0, FY0], [FX1, FY0], [FX1, FY1], [FX0, FY1]], np.float32)
# corner order everywhere: top-left, top-right, bottom-right, bottom-left
BLUE_BGR = (255, 120, 0)
RED_BGR = (0, 80, 255)


BALL_BGR = (60, 210, 240)      # FIRST AGE yellow ball, in BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ball_grid(img, x0, y0, x1, y1, r=7, border=None):
    """A packed block of game balls, like the centre nest and the corner depots."""
    if border is not None:
        cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), border, 3)
    step = int(r * 2.05)
    for yy in range(y0 + r, y1 - r + 1, step):
        for xx in range(x0 + r, x1 - r + 1, step):
            cv2.circle(img, (xx, yy), r, BALL_BGR, -1)
            cv2.circle(img, (xx, yy), r, (40, 160, 190), 1)


def draw_field(blue_left=False):
    """Top-down FIRST AGE field, drawn to match the game diagram.

    Left wall stations run 1->3 top to bottom; the right wall mirrors it (3->1), the
    way the real field is laid out. `blue_left` flips which alliance is on which
    side -- set it to match the broadcast you calibrated against, NOT the manual.
    """
    img = np.full((CH, CW, 3), 252, np.uint8)
    fw, fh = FX1 - FX0, FY1 - FY0
    cx = (FX0 + FX1) // 2

    lcol, rcol = (BLUE_BGR, RED_BGR) if blue_left else (RED_BGR, BLUE_BGR)
    lname, rname = ("B", "R") if blue_left else ("R", "B")

    cv2.rectangle(img, (FX0, FY0), (FX1, FY1), (255, 255, 255), -1)
    cv2.line(img, (cx, FY0), (cx, FY1), (205, 205, 205), 2)

    # --- hub structures: a coloured bar either side of centre, tower in the middle
    for frac, col in ((0.30, lcol), (0.70, rcol)):
        hx = int(FX0 + fw * frac)
        bw = int(fw * 0.037)
        cv2.rectangle(img, (hx - bw, FY0 + int(fh * 0.16)),
                      (hx + bw, FY1 - int(fh * 0.16)), col, -1)
        # feed rails running off both ends of the bar
        cv2.line(img, (hx, FY0 - MARGIN_Y // 2), (hx, FY1 + MARGIN_Y // 2), col, 7)
        cv2.line(img, (hx - bw - 8, FY0 + int(fh * 0.02)),
                 (hx - bw - 8, FY1 - int(fh * 0.02)), col, 3)
        cv2.line(img, (hx + bw + 8, FY0 + int(fh * 0.02)),
                 (hx + bw + 8, FY1 - int(fh * 0.02)), col, 3)
        # scoring tower at the bar's midpoint
        cy = (FY0 + FY1) // 2
        s = int(fh * 0.10)
        cv2.rectangle(img, (hx - s, cy - s), (hx + s, cy + s), (245, 245, 245), -1)
        cv2.rectangle(img, (hx - s, cy - s), (hx + s, cy + s), (150, 150, 150), 3)
        hexa = np.array([[hx + int(s * 0.62 * np.cos(a)), cy + int(s * 0.62 * np.sin(a))]
                         for a in np.linspace(0, 2 * np.pi, 7)[:-1]], np.int32)
        cv2.fillPoly(img, [hexa], (55, 55, 55))

    # --- central ball nest
    nx0, nx1 = int(FX0 + fw * 0.445), int(FX0 + fw * 0.555)
    _ball_grid(img, nx0, FY0 + int(fh * 0.20), nx1, FY1 - int(fh * 0.20))

    # --- corner depots + the angled return structures near each alliance wall
    dw, dh = int(fw * 0.045), int(fh * 0.16)
    _ball_grid(img, FX0 + 8, FY0 + int(fh * 0.17), FX0 + 8 + dw,
               FY0 + int(fh * 0.17) + dh, r=6, border=lcol)
    _ball_grid(img, FX1 - 8 - dw, FY1 - int(fh * 0.17) - dh, FX1 - 8,
               FY1 - int(fh * 0.17), r=6, border=rcol)
    for x0, x1 in ((FX0 + 4, FX0 + int(fw * 0.07)), (FX1 - int(fw * 0.07), FX1 - 4)):
        y0, y1 = FY0 + int(fh * 0.46), FY0 + int(fh * 0.62)
        cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), -1)

    # --- driver stations: left wall 1->3 downward, right wall mirrored 3->1
    for i, fy in enumerate((0.22, 0.47, 0.72)):
        y = int(FY0 + fh * fy)
        pad = int(fh * 0.17)
        cv2.rectangle(img, (FX0 - 46, y - pad // 2), (FX0 - 8, y + pad // 2),
                      (225, 225, 225), -1)
        cv2.putText(img, f"{lname}{i+1}", (FX0 - 92, y + 7), FONT, 0.72, lcol, 2)
        cv2.rectangle(img, (FX1 + 8, y - pad // 2), (FX1 + 46, y + pad // 2),
                      (225, 225, 225), -1)
        cv2.putText(img, f"{rname}{3-i}", (FX1 + 54, y + 7), FONT, 0.72, rcol, 2)

    cv2.rectangle(img, (FX0, FY0), (FX1, FY1), (25, 25, 25), 4)
    return img


# --- fisheye-absorbing polynomial field map -------------------------------
# A 4-corner homography only flattens a pinhole view. This camera is barrel-
# distorted, so we fit a 2nd-order polynomial (x,y)->(X,Y) from a 3x3 grid of known
# field landmarks; the extra middle/edge points capture the bow a homography can't.
_CXM, _CYM = (FX0 + FX1) // 2, (FY0 + FY1) // 2
POLY_LANDMARKS = [
    ("TL", (FX0, FY0)), ("TM", (_CXM, FY0)), ("TR", (FX1, FY0)),
    ("ML", (FX0, _CYM)), ("C", (_CXM, _CYM)), ("MR", (FX1, _CYM)),
    ("BL", (FX0, FY1)), ("BM", (_CXM, FY1)), ("BR", (FX1, FY1)),
]


def _design(pts, degree):
    p = np.asarray(pts, float)
    x, y = p[:, 0], p[:, 1]
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append((x ** (d - i)) * (y ** i))
    return np.stack(cols, 1)


def fit_fieldmap(src_pts, degree=2):
    """src_pts: camera CROP points in POLY_LANDMARKS order -> polynomial coeffs."""
    dst = np.array([t for _, t in POLY_LANDMARKS], float)
    A = _design(src_pts, degree)
    cx = np.linalg.lstsq(A, dst[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(A, dst[:, 1], rcond=None)[0]
    return {"cx": cx.tolist(), "cy": cy.tolist(), "degree": degree,
            "src_pts": [list(map(float, p)) for p in src_pts]}


def apply_fieldmap(fm, pts):
    A = _design(pts, fm["degree"])
    return np.stack([A @ np.array(fm["cx"]), A @ np.array(fm["cy"])], 1)


def save_fieldmap(src_pts, degree=2):
    fm = fit_fieldmap(src_pts, degree)
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    calib["fieldmap"] = fm
    with open(CALIB, "w") as f:
        json.dump(calib, f, indent=2)
    return fm


def load_mapper():
    """A callable pts->top-down: polynomial field map if calibrated, else the
    4-corner homography, else None."""
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    fm = calib.get("fieldmap")
    if fm:
        # a hand-calibrated fit may be stored as a HOMOGRAPHY (the carpet is a
        # plane, and it extrapolates sanely where no landmark could be clicked)
        # rather than as polynomial coefficients.
        if fm.get("kind") == "homography" or "H" in fm:
            H = np.array(fm["H"], np.float32)
            return lambda pts: project(H, pts)
        return lambda pts: apply_fieldmap(fm, pts)
    tdm = calib.get("topdown")
    if tdm:
        H = np.array(tdm["H"], np.float32)
        return lambda pts: project(H, pts)
    return None


def save_homography(src_corners):
    """src_corners: 4 camera CROP points (TL,TR,BR,BL) -> save H into calib.json."""
    src = np.array(src_corners, np.float32)
    H = cv2.getPerspectiveTransform(src, FIELD_CORNERS)
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    calib["topdown"] = {"H": H.tolist(), "src_corners": src.tolist(),
                        "canvas": [CW, CH], "field": [FX0, FY0, FX1, FY1]}
    with open(CALIB, "w") as f:
        json.dump(calib, f, indent=2)
    return H


def load_homography():
    if not os.path.exists(CALIB):
        return None
    td = json.load(open(CALIB)).get("topdown")
    return np.array(td["H"], np.float32) if td else None


def project(H, pts):
    """Map camera CROP points [[x,y],...] to top-down canvas coords."""
    a = np.array(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(a, H).reshape(-1, 2)
