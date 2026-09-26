"""
Per-frame robot TRAJECTORY logger (foundation for auto-path analysis).

Runs the trained YOLO robot detector + ByteTrack across a match video and writes
every robot's ground position each frame. Unlike detector_estimator.py (which only
keeps a cumulative activity number), this keeps the whole path.

Runs on the Apple-Silicon GPU (MPS) if available -- this is the one pipeline in the
project that actually uses the GPU (YOLO inference), unlike the CPU OCR estimator.

Output per match -> results/<match>/:
  trajectory.csv : frame,t,tid,alliance,cx,cy,conf   (cx,cy = ground point in CROP px)
  tracks.json    : per track id -> voted alliance, first/last frame, n frames

cx,cy is the bottom-center of the robot box (where it touches the floor) in the
CROPPED frame's pixel space -- the right point to push through a field homography.

  python trajectories.py --input videos/2026hop_qm1.mp4 --outdir results/2026hop_qm1
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from boxes import CROP  # legacy 1280x720 fallback; field_crop() is the live path


def field_crop(shape):
    """Field region for THIS frame, as pixels -- resolution independent.

    Two input shapes exist now: a full broadcast frame (16:9), and the pre-split
    top view from preprocess (much wider than tall, scoreboard on top, lower
    stitched angle already removed). Same field, different framing, so pick by
    aspect instead of hard-coding pixels.
    """
    h, w = shape[:2]
    if w / float(h) > 2.2:  # pre-split top view (e.g. 1920x712)
        return 0, int(h * 0.211), w, int(h * 0.99)
    return 0, int(h * 0.139), w, int(h * 0.653)  # full broadcast frame


BLUE_BGR = (255, 120, 0)
RED_BGR = (0, 80, 255)

# Alliance from BUMPER COLOR (physical truth), not the YOLO class (which flips).
# HSV ranges; red wraps hue 0/180 so it needs two. Start-phase frames vote heavier
# (robots lined up, bumpers un-occluded + un-blurred = cleanest color).
# Arena is dimly lit, so the saturation/value floors sit low. That's only safe
# because static field structure is subtracted (see static_color_masks) -- without
# it, loose floors would flood in tower panels and ramps.
BLUE_BUMPER = [((100, 70, 40), (132, 255, 255))]
RED_BUMPER = [((0, 70, 40), (10, 255, 255)), ((166, 70, 40), (180, 255, 255))]
BUMPER_MIN_PX = 20
LAB_MIN_CHROMA = 5.0  # min mean chroma to trust the LAB fallback
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def enhance(bgr):
    """Lift the dim arena: CLAHE on L in LAB. Boosts local contrast so dark bumpers
    gain usable chroma, without blowing out the bright ball pile."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([_CLAHE.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def lab_alliance(bgr):
    """Fallback for frames too dim for the HSV masks: mean chroma direction.
    a* above neutral = red, b* below neutral = blue. Relative, so it degrades
    gracefully in low light instead of falling off a threshold cliff."""
    if bgr is None or bgr.size == 0:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    redness = float(lab[:, :, 1].mean()) - 128.0
    blueness = 128.0 - float(lab[:, :, 2].mean())
    if max(redness, blueness) < LAB_MIN_CHROMA:
        return None
    return "red" if redness > blueness else "blue"


START_PHASE_SEC = 6.0  # frames before this vote x3 (lineup, clearest bumpers)
START_WEIGHT = 3
# Bumpers sit at FLOOR level. Sampling the whole lower box swallows field structure
# behind the robot (the hub tower's blue screens, the red ramps) and flips the call,
# so take only the bottom slice and inset the sides.
BUMPER_STRIP_TOP = 0.70  # start the strip this far down the box (bottom 30%)
BUMPER_INSET = 0.12  # trim this fraction off each side


def bumper_strip(img, box):
    """The floor-level bumper band of a robot box, inset from the sides."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    ty = y1 + int(h * BUMPER_STRIP_TOP)
    ix = int(w * BUMPER_INSET)
    return img[max(0, ty) : max(0, y2), max(0, x1 + ix) : max(0, x2 - ix)]


def color_mask(hsv, ranges):
    m = None
    for lo, hi in ranges:
        r = cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = r if m is None else cv2.bitwise_or(m, r)
    return m


def median_frame(video, crop, n=40):
    """Median of n frames spread across the match. The camera is fixed, so moving
    robots/balls average away and what's left is the empty field."""
    x1, y1, x2, y2 = crop
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f[y1:y2, x1:x2])
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def static_color_masks_from(med):
    """Blue/red masks of the STATIC field -- hub tower panels, ramps, walls.
    Anything blue/red in the empty field is furniture, never a bumper, so we
    subtract it before calling an alliance. Without this, a red robot parked in
    front of the tower's blue panel reads as blue."""
    if med is None:
        return None, None
    hsv = cv2.cvtColor(enhance(med), cv2.COLOR_BGR2HSV)
    k = np.ones((5, 5), np.uint8)
    return (cv2.dilate(color_mask(hsv, BLUE_BUMPER), k), cv2.dilate(color_mask(hsv, RED_BUMPER), k))


def static_color_masks(video, crop, n=40):
    return static_color_masks_from(median_frame(video, crop, n))


def static_fp_boxes(model, med, conf=0.25, imgsz=960, device="cpu"):
    """Boxes the detector fires on the EMPTY field. There are no robots in the
    median frame, so every one of these is a structure false positive (the hub
    towers especially). Live detections overlapping them get dropped."""
    if med is None:
        return []
    res = model.predict(
        med, conf=conf, imgsz=imgsz, agnostic_nms=True, verbose=False, device=device
    )[0]
    if res.boxes is None:
        return []
    return [tuple(float(v) for v in b.xyxy[0]) for b in res.boxes]


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def is_static_fp(box, fp_boxes, thresh=0.45):
    return any(_iou(box, f) >= thresh for f in fp_boxes)


def frame_bumper_masks(crop, bg_b=None, bg_r=None):
    """Per-frame blue/red masks (contrast-lifted, static field structure removed).
    Returns (blue_mask, red_mask, enhanced_crop) -- the enhanced crop is reused for
    the LAB fallback so the lift isn't computed twice."""
    enh = enhance(crop)
    hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
    bm, rm = color_mask(hsv, BLUE_BUMPER), color_mask(hsv, RED_BUMPER)
    if bg_b is not None:
        bm = cv2.bitwise_and(bm, cv2.bitwise_not(bg_b))
        rm = cv2.bitwise_and(rm, cv2.bitwise_not(bg_r))
    return bm, rm, enh


def alliance_from_masks(bm, rm, box):
    """Alliance from bumper-band pixel counts in pre-computed masks."""
    sb, sr = bumper_strip(bm, box), bumper_strip(rm, box)
    nb = int(cv2.countNonZero(sb)) if sb.size else 0
    nr = int(cv2.countNonZero(sr)) if sr.size else 0
    if nb + nr < BUMPER_MIN_PX:
        return None
    return "blue" if nb > nr else "red"


def bumper_alliance(bgr):
    """Blue/red from bumper pixels in a robot crop, or None if too little color."""
    if bgr is None or bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b = sum(
        cv2.countNonZero(cv2.inRange(hsv, np.array(lo), np.array(hi))) for lo, hi in BLUE_BUMPER
    )
    r = sum(cv2.countNonZero(cv2.inRange(hsv, np.array(lo), np.array(hi))) for lo, hi in RED_BUMPER)
    if b + r < BUMPER_MIN_PX:
        return None
    return "blue" if b > r else "red"


def _pick_device(requested):
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# The FRC-tuned ByteTrack config (track_buffer 180 vs stock 30) was written for
# this footage but never actually passed in, so every run used ultralytics stock.
# Measured on 2026hop_qm27: 228 -> 162 track ids for the same 6 robots, with
# identical detection recall. Fewer spurious ids, nothing lost.
FRC_TRACKER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bytetrack_frc.yaml"
)
DEFAULT_TRACKER = FRC_TRACKER if os.path.exists(FRC_TRACKER) else "bytetrack.yaml"


def run(
    input_path,
    outdir,
    weights,
    conf=0.25,
    imgsz=960,
    stride=1,
    tracker=None,
    device="auto",
    write_video=False,
    max_frames=None,
):
    tracker = tracker or DEFAULT_TRACKER
    from ultralytics import YOLO

    os.makedirs(outdir, exist_ok=True)
    device = _pick_device(device)
    model = YOLO(weights)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise OSError(f"Could not open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    probe = cv2.VideoCapture(input_path)
    _ok, _f = probe.read()
    probe.release()
    x1, y1, x2, y2 = field_crop(_f.shape) if _ok else CROP
    cw, ch = x2 - x1, y2 - y1

    writer = None
    if write_video:
        writer = cv2.VideoWriter(
            os.path.join(outdir, "trajectory.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / max(1, stride),
            (cw, ch),
        )

    rows = []
    votes = defaultdict(lambda: [0, 0])  # tid -> [blue, red]
    span = {}  # tid -> [first_frame, last_frame, n]
    med = median_frame(input_path, (x1, y1, x2, y2))  # the empty field
    bg_b, bg_r = static_color_masks_from(med)  # structure colour to ignore
    fp_boxes = static_fp_boxes(model, med, conf, imgsz, device)  # structure detections

    frame_idx = 0
    processed = 0
    while True:
        if max_frames is not None and processed >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if stride > 1 and frame_idx % stride != 0:
            frame_idx += 1
            continue
        crop = frame[y1:y2, x1:x2]
        bm, rm, enh = frame_bumper_masks(crop, bg_b, bg_r)
        res = model.track(
            crop,
            conf=conf,
            imgsz=imgsz,
            agnostic_nms=True,
            tracker=tracker,
            persist=True,
            verbose=False,
            device=device,
        )[0]
        t = frame_idx / fps
        if res.boxes is not None and res.boxes.id is not None:
            for b in res.boxes:
                tid = int(b.id)
                cls = int(b.cls)
                bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0]]
                # alliance from BUMPER COLOR at floor level, static field structure
                # removed; fall back to the YOLO class only if too little colour.
                box = (bx1, by1, bx2, by2)
                if is_static_fp(box, fp_boxes):
                    continue  # hub tower / structure, not a robot
                alliance = (
                    alliance_from_masks(bm, rm, box)
                    or lab_alliance(bumper_strip(enh, box))  # dim-frame fallback
                    or ("blue" if cls == 0 else "red")
                )  # last resort
                w = START_WEIGHT if t < START_PHASE_SEC else 1
                votes[tid][0 if alliance == "blue" else 1] += w
                cx = (bx1 + bx2) / 2.0  # ground point: bottom-center of the box
                cy = by2
                rows.append(
                    (
                        frame_idx,
                        round(t, 3),
                        tid,
                        alliance,
                        round(cx, 1),
                        round(cy, 1),
                        round(float(b.conf), 2),
                    )
                )
                if tid not in span:
                    span[tid] = [frame_idx, frame_idx, 0]
                span[tid][1] = frame_idx
                span[tid][2] += 1
                if writer is not None:
                    col = BLUE_BGR if alliance == "blue" else RED_BGR
                    cv2.rectangle(crop, (int(bx1), int(by1)), (int(bx2), int(by2)), col, 2)
                    cv2.circle(crop, (int(cx), int(cy)), 4, col, -1)
                    cv2.putText(
                        crop,
                        f"{alliance[0].upper()}#{tid}",
                        (int(bx1), int(by1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        col,
                        1,
                    )
        if writer is not None:
            writer.write(crop)
        frame_idx += 1
        processed += 1

    cap.release()
    if writer is not None:
        writer.release()

    with open(os.path.join(outdir, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "tid", "alliance", "cx", "cy", "conf"])
        w.writerows(rows)
    tracks = {
        str(tid): {
            "alliance": "blue" if v[0] >= v[1] else "red",
            "first": span[tid][0],
            "last": span[tid][1],
            "n": span[tid][2],
        }
        for tid, v in votes.items()
    }
    with open(os.path.join(outdir, "tracks.json"), "w") as f:
        # the crop ACTUALLY used, from field_crop() -- not the legacy 1280x720 CROP
        # constant, which this used to record and which no longer matches anything.
        json.dump(
            {
                "device": device,
                "fps": fps,
                "stride": stride,
                "crop": [x1, y1, x2, y2],
                "tracks": tracks,
            },
            f,
            indent=2,
        )
    print(
        f"  {os.path.basename(input_path)}: {len(rows)} points, "
        f"{len(tracks)} tracks, device={device}"
    )
    return {"points": len(rows), "tracks": len(tracks), "device": device}


def main():
    p = argparse.ArgumentParser(description="Log per-frame robot trajectories (YOLO+ByteTrack).")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--weights",
        default=None,
        help="detector weights (default: newest runs/detect/*/weights/best.pt)",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--tracker", default=DEFAULT_TRACKER)
    p.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    p.add_argument("--video", action="store_true", help="also write trajectory.mp4")
    p.add_argument("--max-frames", type=int)
    args = p.parse_args()

    weights = args.weights or _newest_weights()
    run(
        args.input,
        args.outdir,
        weights,
        conf=args.conf,
        imgsz=args.imgsz,
        stride=args.stride,
        tracker=args.tracker,
        device=args.device,
        write_video=args.video,
        max_frames=args.max_frames,
    )


def _newest_weights():
    import glob

    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = glob.glob(os.path.join(parent, "runs", "detect", "*", "weights", "best.pt"))
    if not cands:
        raise SystemExit("No trained weights found under runs/detect/*/weights/best.pt")
    return max(cands, key=os.path.getmtime)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
