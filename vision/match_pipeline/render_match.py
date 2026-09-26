import json
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
# The classic-CV modules (scoring_estimator, ...) live in vision/classic.
CLASSIC = os.path.join(PARENT, "classic")
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
sys.path.insert(0, CLASSIC)
VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")
TRUTH = os.path.join(HERE, "truth")
CALIB = os.path.join(HERE, "calib.json")

from boxes import CROP  # detector crop (offset added back for full-frame draw)

BLUE_BGR = (255, 120, 0)
RED_BGR = (0, 80, 255)
TRAIL = 40  # trail length (processed frames)

# ===========================================================================
# EDIT ME (PyCharm one-click). CLI args override.
# ===========================================================================
MATCH = "2026hop_qm27"  # match key (looks for videos/<MATCH>.mp4)
INPUT = ""  # or a direct video path (overrides MATCH)
START = 0.0  # clip start, seconds
END = 30.0  # clip end, seconds
STRIDE = 1  # 1 = every frame (smooth); 2 = faster
CLASSIFY = True  # label boxes with TBA team numbers (needs calib + truth)
FULL_FRAME = True  # render whole broadcast (scoreboard visible); False = field crop
CONF = 0.25
IMGSZ = 960
OCR_EVERY = 15  # OCR the scoreboard every N rendered frames (HUD only)
# ===========================================================================


def _weights():
    import glob

    c = glob.glob(os.path.join(PARENT, "runs", "detect", "*", "weights", "best.pt"))
    if not c:
        raise SystemExit("No trained weights under runs/detect/*/weights/best.pt")
    return max(c, key=os.path.getmtime)


def _device():
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _team_map(match):
    """{alliance: [team1,team2,team3]} display numbers, from TBA truth, or None."""
    p = os.path.join(TRUTH, f"{match}.json")
    if not os.path.exists(p):
        return None
    t = json.load(open(p))

    def nums(keys):
        return [k.replace("frc", "") for k in (keys or [])]

    return {"blue": nums(t.get("blue_teams")), "red": nums(t.get("red_teams"))}


def run(
    video,
    out_path,
    start_sec,
    end_sec,
    stride=1,
    classify=True,
    full_frame=True,
    conf=0.25,
    imgsz=960,
    match=None,
):
    from ultralytics import YOLO
    from scoring_estimator import ocr_ball_count, SCOREBOARD
    from trajectories import (
        bumper_strip,
        frame_bumper_masks,
        alliance_from_masks,
        lab_alliance,
        static_color_masks,
    )

    dev = _device()
    model = YOLO(_weights())

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    from trajectories import field_crop

    x1, y1, x2, y2 = field_crop((H, W))  # resolution-aware; CROP is legacy
    start_f = int(start_sec * fps)
    end_f = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    # output geometry. Detections are in CROP pixel space, so drawing on the full
    # frame needs the crop origin (x1,y1) added back; drawing on the crop itself
    # needs no offset.
    if full_frame:
        ow, oh, ox, oy = W, H, x1, y1
    else:
        ow, oh, ox, oy = x2 - x1, y2 - y1, 0, 0
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, stride), (ow, oh)
    )

    bg_b, bg_r = static_color_masks(video, (x1, y1, x2, y2))
    # READ identities (identity.json) outrank rank-order guessing: these came from
    # actually reading bumper numbers in the top + side views.
    read_id = {}
    idp = os.path.join(RESULTS, match or "", "identity.json")
    if classify and os.path.exists(idp):
        read_id = {
            int(k): v["team"]
            for k, v in json.load(open(idp)).get("tracks", {}).items()
            if v.get("team")
        }
    teams = _team_map(match) if classify else None
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else None
    stations = {k: tuple(v) for k, v in calib["stations"].items()} if (calib and classify) else None
    zones = calib.get("scoring_zones", []) if calib else []
    tid_team = {}  # tid -> label string
    first_pos = {}  # tid -> (cx,cy) crop coords, for classify
    votes = defaultdict(lambda: [0, 0])
    trails = defaultdict(lambda: deque(maxlen=TRAIL))

    stable = {"blue": [frozenset(), 0], "red": [frozenset(), 0]}
    board = {"blue": None, "red": None}
    fidx = start_f
    n = 0
    while fidx <= end_f:
        ret, frame = cap.read()
        if not ret:
            break
        if stride > 1 and (fidx - start_f) % stride != 0:
            fidx += 1
            continue
        crop = frame[y1:y2, x1:x2]
        bm, rm, enh = frame_bumper_masks(crop, bg_b, bg_r)
        res = model.track(
            crop,
            conf=conf,
            imgsz=imgsz,
            agnostic_nms=True,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            device=dev,
        )[0]

        canvas = frame if full_frame else crop.copy()
        # draw scoring zones (translucent) so you can see the shoot area. Supports
        # 4-corner polygons (fisheye) and legacy rects.
        for z in zones:
            col = BLUE_BGR if z["alliance"] == "blue" else RED_BGR
            if z.get("poly"):
                pts = np.array([[x + ox, y + oy] for x, y in z["poly"]], np.int32)
            else:
                zx, zy, zw, zh = z["rect"]
                pts = np.array(
                    [
                        [zx + ox, zy + oy],
                        [zx + zw + ox, zy + oy],
                        [zx + zw + ox, zy + zh + oy],
                        [zx + ox, zy + zh + oy],
                    ],
                    np.int32,
                )
            ov = canvas.copy()
            cv2.fillPoly(ov, [pts], col)
            cv2.addWeighted(ov, 0.18, canvas, 0.82, 0, canvas)
            cv2.polylines(canvas, [pts], True, col, 1)
        live = []
        if res.boxes is not None and res.boxes.id is not None:
            for b in res.boxes:
                tid = int(b.id)
                cls = int(b.cls)
                votes[tid][cls] += 1
                bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0]]
                gx, gy = (bx1 + bx2) // 2, by2  # ground point (crop coords)
                first_pos.setdefault(tid, (gx, gy))
                _box = (bx1, by1, bx2, by2)
                al = (
                    alliance_from_masks(bm, rm, _box)
                    or lab_alliance(bumper_strip(enh, _box))
                    or ("blue" if cls == 0 else "red")
                )
                live.append((tid, al, bx1, by1, bx2, by2, gx, gy))
                trails[tid].append((gx + ox, gy + oy))

        # a READ number wins outright; rank-order only fills the gaps
        for tid, *_ in live:
            if tid in read_id and tid not in tid_team:
                tid_team[tid] = f"#{read_id[tid]}"
        if classify and teams and live:
            _assign_teams(live, teams, first_pos, tid_team, stable)

        # draw trails + boxes
        for tid, al, bx1, by1, bx2, by2, gx, gy in live:
            col = BLUE_BGR if al == "blue" else RED_BGR
            pts = list(trails[tid])
            for i in range(1, len(pts)):
                a = int(255 * i / len(pts))
                cv2.line(canvas, pts[i - 1], pts[i], col, 2)
            X1, Y1, X2, Y2 = bx1 + ox, by1 + oy, bx2 + ox, by2 + oy
            cv2.rectangle(canvas, (X1, Y1), (X2, Y2), col, 2)
            label = tid_team.get(tid) or f"{al[0].upper()}#{tid}"
            cv2.rectangle(canvas, (X1, Y1 - 18), (X1 + 9 * len(label), Y1), col, -1)
            cv2.putText(
                canvas, label, (X1 + 2, Y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # scoreboard OCR (HUD only) -- every OCR_EVERY frames; tesseract on every
        # frame would dominate a full-match render.
        if n % OCR_EVERY == 0:
            for a in ("blue", "red"):
                v = ocr_ball_count(frame, a)
                if v is not None and v < 1500:
                    if board[a] is None or v >= board[a] - 3:
                        board[a] = v
        _hud(canvas, match or os.path.basename(video), fidx / fps, board)

        writer.write(canvas)
        fidx += 1
        n += 1

    cap.release()
    writer.release()
    print(f"  rendered {n} frames -> {out_path}  (device={dev})")
    return out_path


def _assign_teams(live, teams, first_pos, tid_team, stable, need=15):
    """Lock team labels only once the SAME three tracks have held for `need`
    consecutive frames.

    Station comes from rank-by-position, which is only meaningful when all three
    robots are present. Assigning off a single frame bakes in whatever transient
    junk was on screen (a half-occluded box, a missed robot) and -- because labels
    are sticky -- that wrong number never gets revisited.
    """
    from auto_paths import ALLIANCE_ORDER

    for al in ("blue", "red"):
        tids = {tid for tid, a, *_ in live if a == al}
        key = frozenset(tids)
        if key == stable[al][0]:
            stable[al][1] += 1
        else:
            stable[al] = [key, 1]
        if len(tids) != 3 or stable[al][1] < need:
            continue
        if any(t in tid_team for t in tids):
            continue  # already locked
        ordered = sorted(
            tids, key=lambda t: first_pos[t][0], reverse=(ALLIANCE_ORDER.get(al) == "rtl")
        )
        for idx, tid in enumerate(ordered):
            if idx < len(teams[al]) and teams[al][idx]:
                tid_team[tid] = f"#{teams[al][idx]}"


def _hud(img, title, t, board):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    mm, ss = int(t) // 60, int(t) % 60
    cv2.putText(
        img,
        f"{title}   t={mm}:{ss:02d}",
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    txt = f"BALLS  blue {board['blue']}   red {board['red']}"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(
        img, txt, (img.shape[1] - tw - 10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )


def main():
    # No CLI. Everything comes from the EDIT ME block at the top of this file.
    # Change MATCH / START / END up there, hit Run.
    video = INPUT or os.path.join(VIDEOS, f"{MATCH}.mp4")
    if not os.path.exists(video):
        raise SystemExit(f"No video: {video}")
    match = MATCH if not INPUT else os.path.splitext(os.path.basename(video))[0]
    out = os.path.join(RESULTS, match, f"clip_{int(START)}-{int(END)}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run(
        video, out, START, END, stride=STRIDE, classify=CLASSIFY, full_frame=FULL_FRAME, match=match
    )
    print(f"\nOpen it:  {out}")


if __name__ == "__main__":
    main()
