"""
AUTO PATH VIEW -- one clean picture of what all six robots did during auto.

Draws each robot's stitched auto path on the empty field (the median frame, so no
robots or balls clutter it), labelled with its team number AND how that number was
決 determined:

  read   green    bumper number actually read off the robot
  elim   cyan     every other robot on the alliance was read, so this one is the
                  only team left -- certain, and owes nothing to station order
  rank   orange   left-to-right station guess. ALLIANCE_ORDER is UNVALIDATED, so
                  treat every orange label as unproven.
  none   grey     not determined

Paths are the STITCHED chain, so a robot whose track id changed mid-auto is one
continuous path instead of several differently-labelled pieces.

PyCharm: set MATCH (or MATCHES) below, hit Run.
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
VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")
TRUTH = os.path.join(HERE, "truth")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
MATCH = "2026hop_qm100"   # single match to draw
MATCHES = []              # or a list of match keys; [] = just MATCH
CONTACT_SHEET = False     # True + MATCHES -> also write one combined grid image
DOTS_EVERY = 12           # progress dot every N path samples (shows speed)
# ===========================================================================

BLUE_BGR = (255, 120, 0)
RED_BGR = (0, 80, 255)
SRC_COL = {"read": (90, 255, 90), "elim": (255, 220, 90),
           "rank": (0, 165, 255), "none": (150, 150, 150)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img, text, org, color=(255, 255, 255), scale=0.55, thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _background(match):
    """Empty field: median over the match, cropped to the field region."""
    import trajectories as tj
    vid = os.path.join(VIDEOS, f"{match}.mp4")
    if not os.path.exists(vid):
        return None
    cap = cv2.VideoCapture(vid)
    ok, f0 = cap.read()
    cap.release()
    if not ok:
        return None
    crop = tj.field_crop(f0.shape)
    med = tj.median_frame(vid, crop)
    if med is None:
        x1, y1, x2, y2 = crop
        med = f0[y1:y2, x1:x2]
    return (med * 0.55).astype(np.uint8)      # darkened so paths pop


def render(match, out=None, dots_every=DOTS_EVERY):
    ap_path = os.path.join(RESULTS, match, "auto_paths.json")
    if not os.path.exists(ap_path):
        print(f"  {match}: no auto_paths.json -- skipped")
        return None
    data = json.load(open(ap_path))
    teams = data.get("teams", {})
    dbg = data.get("debug", {})

    img = _background(match)
    if img is None:
        print(f"  {match}: no video -- skipped")
        return None
    h, w = img.shape[:2]
    panel = 300
    canvas = np.zeros((max(h, 260), w + panel, 3), np.uint8)
    canvas[0:h, 0:w] = img

    counts = {}
    for st in sorted(teams):
        o = teams[st]
        src = o.get("source") or "none"
        counts[src] = counts.get(src, 0) + 1
        pts = [(int(cx), int(cy)) for _t, cx, cy in o.get("path", [])]
        if len(pts) < 2:
            continue
        base = BLUE_BGR if o["alliance"] == "blue" else RED_BGR
        arr = np.array(pts, np.int32)
        cv2.polylines(canvas[0:h, 0:w], [arr], False, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.polylines(canvas[0:h, 0:w], [arr], False, base, 3, cv2.LINE_AA)
        for i in range(0, len(pts), max(1, dots_every)):
            cv2.circle(canvas, pts[i], 3, (255, 255, 255), -1)
        # start marker + direction at the end
        cv2.circle(canvas, pts[0], 8, (255, 255, 255), -1)
        cv2.circle(canvas, pts[0], 8, base, 2)
        if len(pts) >= 6:
            cv2.arrowedLine(canvas, pts[-6], pts[-1], base, 3, cv2.LINE_AA,
                            tipLength=0.5)
        label = f"{str(o.get('team')).replace('frc', '#')} [{src}]"
        _put(canvas, label, (pts[0][0] + 12, pts[0][1] - 12),
             SRC_COL.get(src, (255, 255, 255)), 0.62, 2)

    # ---- side panel -------------------------------------------------------
    cv2.rectangle(canvas, (w, 0), (w + panel, canvas.shape[0]), (22, 22, 22), -1)
    y = 30
    _put(canvas, match, (w + 14, y), (255, 255, 255), 0.66, 2); y += 26
    _put(canvas, f"auto frames {dbg.get('start_frame')}-{dbg.get('end_frame')}",
         (w + 14, y), (170, 170, 170), 0.46); y += 30

    truth_p = os.path.join(TRUTH, f"{match}.json")
    if os.path.exists(truth_p):
        tr = json.load(open(truth_p))
        for al, col in (("blue", BLUE_BGR), ("red", RED_BGR)):
            roster = " ".join(t.replace("frc", "#") for t in (tr.get(f"{al}_teams") or []))
            _put(canvas, f"TBA {al}: {roster}", (w + 14, y), col, 0.46); y += 20
        y += 10

    _put(canvas, "ASSIGNED", (w + 14, y), (255, 255, 255), 0.52, 1); y += 22
    for st in sorted(teams):
        o = teams[st]
        src = o.get("source") or "none"
        n = len(o.get("tids") or [o.get("tid")])
        _put(canvas, f"{st:6s} {str(o.get('team')).replace('frc', '#'):8s} "
                     f"{src:5s} ({n} frag)", (w + 14, y),
             SRC_COL.get(src, (255, 255, 255)), 0.46); y += 19
    y += 14
    _put(canvas, "HOW DETERMINED", (w + 14, y), (255, 255, 255), 0.52, 1); y += 22
    for k in ("read", "elim", "rank", "none"):
        if counts.get(k):
            _put(canvas, f"{k:5s} {counts[k]}", (w + 14, y), SRC_COL[k], 0.5); y += 19
    y += 12
    if counts.get("rank"):
        _put(canvas, "orange = station-order guess,", (w + 14, y), (0, 165, 255), 0.42)
        y += 16
        _put(canvas, "UNVALIDATED. do not trust.", (w + 14, y), (0, 165, 255), 0.42)

    out = out or os.path.join(RESULTS, match, "auto_view.png")
    cv2.imwrite(out, canvas)
    print(f"  {match}: {counts} -> {out}")
    return out


def main():
    ms = MATCHES or [MATCH]
    paths = [p for p in (render(m) for m in ms) if p]
    if CONTACT_SHEET and len(paths) > 1:
        imgs = [cv2.imread(p) for p in paths]
        wmax = max(i.shape[1] for i in imgs)
        scaled = [cv2.resize(i, (wmax, int(i.shape[0] * wmax / i.shape[1])))
                  for i in imgs]
        sheet = np.vstack(scaled)
        out = os.path.join(RESULTS, "auto_view_sheet.png")
        cv2.imwrite(out, sheet)
        print(f"contact sheet -> {out}")


if __name__ == "__main__":
    main()
