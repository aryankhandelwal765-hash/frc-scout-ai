"""
Per-match robot RE-IDENTIFICATION: name tracks by what the robot looks like.

Bumper numbers are read reliably only in auto, where robots are apart and facing the
cameras. A robot looks the same all match though, so the tracks identified in auto (and
wherever else a read landed) become labelled examples, and every other track is named by
its nearest labelled team among the SAME alliance's three robots.

    python reid.py --match 2026hop_qm28 --eval        # leave-one-track-out accuracy
    python reid.py --match 2026hop_qm28               # write reid.json

Embeddings: ImageNet resnet18 (global-pooled) on the whole robot box, L2-normalised,
averaged over the track's sampled frames. Output reid.json maps tid -> {team, margin}.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")
VIDEOS = os.path.join(HERE, "videos")

SAMPLE_EVERY = 30  # video frames between embedded frames
BIND_PX = 120  # a detection must be this close to a logged track point
MIN_BOX = 40  # px; smaller boxes carry no usable appearance
SEED_CONF = 0.6  # identity.json confidence a track needs to be a labelled example
_NET = None


def net():
    global _NET
    if _NET is None:
        import ssl

        import certifi
        import torch
        import torchvision

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
            cafile=certifi.where()
        )
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        m.fc = torch.nn.Identity()
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _NET = (m.eval().to(dev), dev)
    return _NET


def embed(crops):
    import torch

    m, dev = net()
    x = np.stack([cv2.resize(c, (96, 128))[..., ::-1] for c in crops]).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    with torch.no_grad():
        f = m(torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)).cpu().numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def track_embeddings(match, model=None):
    """tid -> (mean unit embedding, n crops, first frame, last frame)"""
    import trajectories as tj

    pts = defaultdict(dict)
    with open(os.path.join(RESULTS, match, "trajectory.csv")) as f:
        for r in csv.DictReader(f):
            pts[int(r["frame"])][int(r["tid"])] = (float(r["cx"]), float(r["cy"]))
    if model is None:
        from ultralytics import YOLO

        model = YOLO(tj._newest_weights())
    frames = sorted(pts)[:: max(1, SAMPLE_EVERY // 2)]
    cap = cv2.VideoCapture(os.path.join(VIDEOS, f"{match}.mp4"))
    feats = defaultdict(list)
    seen = defaultdict(list)
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        x1, y1, x2, y2 = tj.field_crop(frame.shape)
        crop = frame[y1:y2, x1:x2]
        res = model.predict(crop, conf=0.3, imgsz=960, agnostic_nms=True, verbose=False)[0]
        if res.boxes is None:
            continue
        got = []
        for b in res.boxes:
            bx1, by1, bx2, by2 = (int(v) for v in b.xyxy[0])
            if bx2 - bx1 < MIN_BOX or by2 - by1 < MIN_BOX:
                continue
            gx, gy = (bx1 + bx2) / 2, by2
            best = (
                min(((gx - px) ** 2 + (gy - py) ** 2, t) for t, (px, py) in pts[f].items())
                if pts[f]
                else None
            )
            if best and best[0] <= BIND_PX**2:
                got.append((best[1], crop[max(0, by1) : by2, max(0, bx1) : bx2]))
        if got:
            for (t, _c), e in zip(got, embed([c for _t, c in got])):
                feats[t].append(e)
                seen[t].append(f)
    cap.release()
    out = {}
    for t, es in feats.items():
        m = np.mean(es, axis=0)
        out[t] = (m / np.linalg.norm(m), len(es), min(seen[t]), max(seen[t]))
    return out


def classify(emb, labels, alliance_of, teams_of, exclude=(), min_gap=0, spans=None):
    """tid -> (team, margin) using prototypes of the labelled tracks.

    exclude : tids never used as examples (leave-one-out)
    min_gap : frames; also skip labelled examples whose span is within this of the query
    """
    proto = defaultdict(list)
    for t, team in labels.items():
        if t in emb and t not in exclude:
            proto[team].append(emb[t][0])
    out = {}
    return proto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--match", required=True)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--cache")
    args = ap.parse_args()
    import pickle

    if args.cache and os.path.exists(args.cache):
        emb = pickle.load(open(args.cache, "rb"))
    else:
        emb = track_embeddings(args.match)
        if args.cache:
            pickle.dump(emb, open(args.cache, "wb"))
    print(f"{len(emb)} tracks embedded")


if __name__ == "__main__":
    main()
