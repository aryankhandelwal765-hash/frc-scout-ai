"""
Harvest labelled bumper-number crops from matches we already identified.

identify.py names a track after voting many OCR reads, so a track with a strong team
is a free label for EVERY frame of that track, not just the frames OCR happened to
read. This walks each match video, runs the current detector, binds each detection to
the nearest logged track point (same rule as identify.py), and when that track has a
confident team, saves the bumper band with the team number as its label.

    python build_digit_crops.py --every 90 --out ../digit_data
    python build_digit_crops.py --match 2026hop_qm28 --every 30 --out ../digit_data

Output: <out>/crops/<match>_f<frame>_t<tid>.jpg and <out>/labels.csv
        (file,team,match,alliance,box_w,box_h)

Labels are per-TRACK, so some crops show an unreadable bumper (turned away, occluded)
under a valid label. train_digits.py cleans those with its own first-pass model.
"""

import argparse
import csv
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.path.join(os.path.dirname(HERE), "match_pipeline")
sys.path.insert(0, MP)
sys.path.insert(0, os.path.dirname(HERE))

import read_numbers as rn  # noqa: E402

import trajectories as tj  # noqa: E402

RESULTS = os.path.join(MP, "results")
VIDEOS = os.path.join(MP, "videos")
MIN_CONF = 0.9  # identity confidence a track needs to lend its label
MIN_VOTES = 3.0  # and total vote weight on that team
BIND_PX = 120  # detection ground point must be this close to a logged track point
AMBIGUOUS_PX = 60  # two differently-labelled tracks this close = skip the detection


def strong_tracks(match):
    p = os.path.join(RESULTS, match, "identity.json")
    if not os.path.exists(p):
        return {}
    out = {}
    for tid, v in json.load(open(p))["tracks"].items():
        if v.get("source") == "read" and v["confidence"] >= MIN_CONF:
            if v["votes"].get(v["team"], 0) >= MIN_VOTES:
                out[int(tid)] = v["team"]
    return out


def load_points(match):
    pts = {}
    with open(os.path.join(RESULTS, match, "trajectory.csv")) as f:
        for r in csv.DictReader(f):
            pts.setdefault(int(r["frame"]), {})[int(r["tid"])] = (float(r["cx"]), float(r["cy"]))
    return pts


def harvest(match, model, every, out, writer):
    labels = strong_tracks(match)
    vid = os.path.join(VIDEOS, f"{match}.mp4")
    if not labels or not os.path.exists(vid):
        return 0
    pts = load_points(match)
    frames = sorted(pts)[:: max(1, every // 2)]  # trajectory is logged every 2nd frame
    cap = cv2.VideoCapture(vid)
    n = 0
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        cx0, cy0, cx1, cy1 = tj.field_crop(frame.shape)
        crop = frame[cy0:cy1, cx0:cx1]
        res = model.predict(crop, conf=0.3, imgsz=960, agnostic_nms=True, verbose=False)[0]
        if res.boxes is None:
            continue
        near = pts[f]
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            gx, gy = (x1 + x2) / 2, y2
            ranked = sorted(
                ((gx - px) ** 2 + (gy - py) ** 2, tid) for tid, (px, py) in near.items()
            )
            if not ranked or ranked[0][0] > BIND_PX**2:
                continue
            tid = ranked[0][1]
            if tid not in labels:
                continue
            if any(
                tid2 != tid and d2 < AMBIGUOUS_PX**2 and labels.get(tid2) not in (None, labels[tid])
                for d2, tid2 in ranked[1:]
            ):
                continue
            band = rn.bumper_crop(crop, (x1, y1, x2, y2))
            if band.size == 0 or band.shape[0] < 8 or band.shape[1] < 24:
                continue
            name = f"{match}_f{f}_t{tid}.jpg"
            cv2.imwrite(os.path.join(out, "crops", name), band, [cv2.IMWRITE_JPEG_QUALITY, 95])
            writer.writerow(
                [name, labels[tid], match, tj_alliance(match, tid), int(x2 - x1), int(y2 - y1)]
            )
            n += 1
    cap.release()
    return n


_AL = {}


def tj_alliance(match, tid):
    if match not in _AL:
        d = json.load(open(os.path.join(RESULTS, match, "identity.json")))["tracks"]
        _AL[match] = {int(k): v["alliance"] for k, v in d.items()}
    return _AL[match].get(tid, "")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--match")
    p.add_argument("--every", type=int, default=90, help="video frames between samples")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(tj._newest_weights())
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    matches = (
        [args.match]
        if args.match
        else sorted(
            m
            for m in os.listdir(RESULTS)
            if os.path.exists(os.path.join(RESULTS, m, "identity.json"))
            and os.path.exists(os.path.join(VIDEOS, f"{m}.mp4"))
        )
    )
    lp = os.path.join(args.out, "labels.csv")
    new = not os.path.exists(lp)
    with open(lp, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["file", "team", "match", "alliance", "box_w", "box_h"])
        for m in matches:
            n = harvest(m, model, args.every, args.out, w)
            fh.flush()
            print(f"{m}: {n} crops", flush=True)


if __name__ == "__main__":
    main()
