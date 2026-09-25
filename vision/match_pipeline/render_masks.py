"""
Debug render: SHOW the blue/red bumper masks.

Renders the field crop with every pixel the BLUE bumper mask catches painted blue
and every pixel the RED mask catches painted red, over a darkened frame. Each
detected robot is boxed and labelled with its actual pixel counts and the verdict:

    b=812 r=44 -> blue

That tells you exactly why an alliance call is wrong -- e.g. the red mask lighting
up the red goal structure inside a robot's box, or a bumper too dark to register.

Tune BLUE_BUMPER / RED_BUMPER / BUMPER_MIN_PX in trajectories.py from what you see.

PyCharm: set MATCH / START / END, click Run.
"""

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
from boxes import CROP

# ===========================================================================
# EDIT ME. Click Run.
# ===========================================================================
MATCH = "2026hop_qm27"
START = 0.0            # clip start, seconds
END = 25.0             # clip end, seconds
STRIDE = 2
CONF = 0.25
IMGSZ = 960
DIM = 0.30             # how dark the background frame is under the masks
SHOW_ONLY_IN_BOX = False   # True = paint masks only inside robot boxes
# ===========================================================================


def main():
    from ultralytics import YOLO
    import trajectories as tj

    video = os.path.join(VIDEOS, f"{MATCH}.mp4")
    if not os.path.exists(video):
        raise SystemExit(f"No video: {video}")
    out = os.path.join(RESULTS, MATCH, f"masks_{int(START)}-{int(END)}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    model = YOLO(tj._newest_weights())
    dev = tj._pick_device("auto")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    x1, y1, x2, y2 = CROP
    cw, ch = x2 - x1, y2 - y1
    start_f, end_f = int(START * fps), int(END * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(1, STRIDE), (cw, ch))

    print("building static field-structure mask ...")
    bg_b, bg_r = tj.static_color_masks(video, CROP)

    fidx, n = start_f, 0
    while fidx <= end_f:
        ret, frame = cap.read()
        if not ret:
            break
        if STRIDE > 1 and (fidx - start_f) % STRIDE != 0:
            fidx += 1
            continue
        crop = frame[y1:y2, x1:x2]
        bmask, rmask, enh = tj.frame_bumper_masks(crop, bg_b, bg_r)

        res = model.track(crop, conf=CONF, imgsz=IMGSZ, agnostic_nms=True,
                          tracker="bytetrack.yaml", persist=True, verbose=False,
                          device=dev)[0]

        if SHOW_ONLY_IN_BOX:
            keep = np.zeros((ch, cw), np.uint8)
            if res.boxes is not None:
                for b in res.boxes:
                    bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0]]
                    keep[max(0, by1):by2, max(0, bx1):bx2] = 255
            bmask = cv2.bitwise_and(bmask, keep)
            rmask = cv2.bitwise_and(rmask, keep)

        canvas = (crop * DIM).astype(np.uint8)
        canvas[bmask > 0] = (255, 120, 0)     # blue mask hits
        canvas[rmask > 0] = (0, 80, 255)      # red mask hits

        if res.boxes is not None and res.boxes.id is not None:
            for b in res.boxes:
                tid = int(b.id); cls = int(b.cls)
                bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0]]
                box = (bx1, by1, bx2, by2)
                sb = tj.bumper_strip(bmask, box)
                sr = tj.bumper_strip(rmask, box)
                nb = int(cv2.countNonZero(sb)) if sb.size else 0
                nr = int(cv2.countNonZero(sr)) if sr.size else 0
                verdict = (tj.alliance_from_masks(bmask, rmask, box)
                           or tj.lab_alliance(tj.bumper_strip(enh, box)))
                # outline the exact strip being sampled
                sy = by1 + int((by2 - by1) * tj.BUMPER_STRIP_TOP)
                sx = int((bx2 - bx1) * tj.BUMPER_INSET)
                cv2.rectangle(canvas, (bx1 + sx, sy), (bx2 - sx, by2), (255, 255, 255), 1)
                col = (255, 120, 0) if verdict == "blue" else (
                    (0, 80, 255) if verdict == "red" else (200, 200, 200))
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), col, 2)
                txt = f"#{tid} b={nb} r={nr} -> {verdict or 'none'} (cls={'B' if cls==0 else 'R'})"
                cv2.rectangle(canvas, (bx1, by1 - 16), (bx1 + 7 * len(txt), by1), (0, 0, 0), -1)
                cv2.putText(canvas, txt, (bx1 + 2, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.rectangle(canvas, (0, 0), (cw, 22), (0, 0, 0), -1)
        cv2.putText(canvas, f"{MATCH}  BUMPER MASKS  blue=painted blue  red=painted red"
                            f"   t={fidx/fps:.1f}s", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(canvas)
        fidx += 1
        n += 1

    cap.release()
    writer.release()
    print(f"rendered {n} frames -> {out}  (device={dev})")
    print("Tune BLUE_BUMPER / RED_BUMPER / BUMPER_MIN_PX in trajectories.py")


if __name__ == "__main__":
    main()
