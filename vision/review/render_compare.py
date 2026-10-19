"""
Render ONE frame three ways, stacked: BEFORE (old detector, no filters), AFTER (new
detector + today's filters), and the HUMAN labels, with per-panel scores vs the human.

    python render_compare.py --image dataset_v3/images/val/2026hop_qm103_f000778.jpg \
        --human dataset_v3/labels/val/2026hop_qm103_f000778.txt \
        --old runs/detect/frc_robots-2/weights/best.pt \
        --new runs/experiments/frc_robots_v3/weights/best.pt --out compare.png

BEFORE = old nano detector at 960px, conf 0.25, raw boxes, bumper colour or blue by default.
AFTER  = new detector at 1280px, nested-duplicate merge, person veto, colour elimination.
(The per-video static-structure filter needs the whole video, so it is not in this render.)
"""

import argparse
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import prelabel as pl  # noqa: E402

rf = pl.rf
tj = pl.tj
COLORS = {0: (255, 120, 0), 1: (0, 80, 255)}


def read_yolo(path, w, h):
    out = []
    if os.path.exists(path):
        for ln in open(path):
            q = ln.split()
            if len(q) == 5:
                c, cx, cy, bw, bh = int(q[0]), *(float(v) for v in q[1:])
                out.append(
                    (
                        c,
                        (
                            (cx - bw / 2) * w,
                            (cy - bh / 2) * h,
                            (cx + bw / 2) * w,
                            (cy + bh / 2) * h,
                        ),
                    )
                )
    return out


def score(pred, gold, thr=0.5):
    used, tp = set(), 0
    for c, b in pred:
        best, bj = 0.0, None
        for j, (gc, gb) in enumerate(gold):
            if j in used or gc != c:
                continue
            v = tj._iou(b, gb)
            if v > best:
                best, bj = v, j
        if bj is not None and best >= thr:
            used.add(bj)
            tp += 1
    return tp, len(pred) - tp, len(gold) - tp


def colour_boxes(crop, boxes):
    bm, rm, _ = tj.frame_bumper_masks(crop, None, None)
    return [tj.alliance_from_masks(bm, rm, b) for b in boxes]


def draw(img, boxes, caption):
    out = img.copy()
    for c, b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[c], 3)
    bar = 46
    canvas = cv2.copyMakeBorder(out, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(canvas, caption, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--human", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--person-model", default=os.path.join(os.path.dirname(HERE), "yolov8n.pt"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from ultralytics import YOLO

    device = pl.pick_device("auto")
    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    gold = read_yolo(args.human, w, h)

    # BEFORE
    old = YOLO(args.old)
    raw = [b for b, _c in pl.detect(old, img, args.conf, 960, device)]
    before = [(0 if a in (None, "blue") else 1, b) for a, b in zip(colour_boxes(img, raw), raw)]

    # AFTER
    new = YOLO(args.new)
    dets = pl.merge([("ft1280", pl.detect(new, img, args.conf, 1280, device))])
    dets = rf.merge_nested(dets)
    people = pl.detect_people(YOLO(args.person_model), img, 0.35, device)
    kept, _gone = rf.veto_people([d["box"] for d in dets], people)
    kept = set(kept)
    dets = [d for d in dets if d["box"] in kept]
    boxes = [d["box"] for d in dets]
    colours, _how = rf.eliminate_alliances(colour_boxes(img, boxes))
    after = [(0 if c in (None, "blue") else 1, b) for c, b in zip(colours, boxes)]

    rows = []
    for name, preds in (
        ("BEFORE  old nano detector, raw boxes", before),
        ("AFTER   new detector + person veto + duplicate merge + colour elimination", after),
        ("HUMAN   corrected labels", gold),
    ):
        tp, fp, fn = score(preds, gold)
        tag = (
            ""
            if name.startswith("HUMAN")
            else f"   {len(preds)} boxes: {tp} right, {fp} wrong/extra, {fn} missed"
        )
        rows.append(draw(img, preds, name + tag))
    cv2.imwrite(args.out, cv2.vconcat(rows))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
