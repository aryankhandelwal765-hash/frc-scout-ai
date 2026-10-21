"""
Harvest labelled whole-robot crops across every identified match.

A robot looks the same all event, so crops of a team from OTHER matches are the best
examples for naming a track here. A track with a bumper read (identity.json, confidence
>= MIN_CONF) lends its team to every sampled frame of that track.

    python harvest_robots.py --out ../reid_data --every 30

Output: <out>/crops/<match>_f<frame>_t<tid>.jpg (whole robot box, 112x144) and
        <out>/labels.csv (file,team,match,tid,frame,alliance)
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
MIN_CONF = 0.6
BIND_PX = 120
MIN_BOX = 40
CROP_WH = (112, 144)


def labelled_tracks(match):
    p = os.path.join(RESULTS, match, "identity.json")
    if not os.path.exists(p):
        return {}
    alliance_of = {}
    for a in ("blue", "red"):
        for t in rn.alliance_candidates(match, a):
            alliance_of[t] = a
    out = {}
    for tid, v in json.load(open(p))["tracks"].items():
        if v.get("source") == "read" and v["confidence"] >= MIN_CONF and v["team"] in alliance_of:
            out[int(tid)] = (v["team"], alliance_of[v["team"]])
    return out


def harvest(match, model, every, out, writer):
    labels = labelled_tracks(match)
    vid = os.path.join(VIDEOS, f"{match}.mp4")
    if not labels or not os.path.exists(vid):
        return 0
    pts = {}
    with open(os.path.join(RESULTS, match, "trajectory.csv")) as f:
        for r in csv.DictReader(f):
            pts.setdefault(int(r["frame"]), {})[int(r["tid"])] = (float(r["cx"]), float(r["cy"]))
    cap = cv2.VideoCapture(vid)
    n = 0
    for f in sorted(pts)[:: max(1, every // 2)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        x1, y1, x2, y2 = tj.field_crop(frame.shape)
        crop = frame[y1:y2, x1:x2]
        res = model.predict(crop, conf=0.3, imgsz=960, agnostic_nms=True, verbose=False)[0]
        for b in res.boxes if res.boxes is not None else []:
            bx1, by1, bx2, by2 = (max(0, int(v)) for v in b.xyxy[0])
            if bx2 - bx1 < MIN_BOX or by2 - by1 < MIN_BOX or not pts[f]:
                continue
            gx, gy = (bx1 + bx2) / 2, by2
            d, tid = min(((gx - px) ** 2 + (gy - py) ** 2, t) for t, (px, py) in pts[f].items())
            if d > BIND_PX**2 or tid not in labels:
                continue
            name = f"{match}_f{f}_t{tid}.jpg"
            cv2.imwrite(
                os.path.join(out, "crops", name),
                cv2.resize(crop[by1:by2, bx1:bx2], CROP_WH),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            writer.writerow([name, labels[tid][0], match, tid, f, labels[tid][1]])
            n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--match")
    args = ap.parse_args()
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
            w.writerow(["file", "team", "match", "tid", "frame", "alliance"])
        for m in matches:
            print(f"{m}: {harvest(m, model, args.every, args.out, w)} crops", flush=True)
            fh.flush()


if __name__ == "__main__":
    main()
