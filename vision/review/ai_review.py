"""
AI review of prelabeled frames: a vision model says keep / delete / recolour per box and
points out missed robots; a human still draws anything new.

    python ai_review.py prepare --pre dataset_v2 --frames frames.txt --out ai_review/in
    # ... a model reads ai_review/in/<frame>.png + .boxes.json, writes
    #     ai_review/verdicts_<model>/<frame>.json ...
    python ai_review.py score --pre dataset_v2 --truth <human-corrected labels dir> \
        --verdicts ai_review/verdicts_sonnet --frames frames.txt

Verdict file schema (one per frame):
  {"boxes": [{"id": 0, "verdict": "keep"|"delete"|"recolor", "colour": "blue"|"red"},
             ...one entry per numbered box...],
   "missed": [{"colour": "blue"|"red"|"unknown", "box": [x1, y1, x2, y2]}],  # 0..1 fractions
   "unsure": false}

`prepare` draws numbered boxes on a copy of the frame; the model never sees the labels
file, so scoring against human labels is a fair test.
"""

import argparse
import json
import os

import cv2

CLASS_NAMES = {0: "blue", 1: "red"}
BOX_COL = {0: (255, 120, 0), 1: (0, 80, 255)}


def read_labels(path, w, h):
    boxes = []
    if not os.path.exists(path):
        return boxes
    for line in open(path):
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
        boxes.append(
            {
                "cls": c,
                "box": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
            }
        )
    return boxes


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def frame_names(path):
    return [ln.strip() for ln in open(path) if ln.strip()]


def prepare(args):
    os.makedirs(args.out, exist_ok=True)
    for name in frame_names(args.frames):
        img = cv2.imread(os.path.join(args.pre, "images", name + ".jpg"))
        h, w = img.shape[:2]
        boxes = read_labels(os.path.join(args.pre, "labels", name + ".txt"), w, h)
        out = img.copy()
        meta = []
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in b["box"])
            col = BOX_COL[b["cls"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, max(th + 8, y1)), col, -1)
            cv2.putText(
                out,
                label,
                (x1 + 4, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            meta.append(
                {
                    "id": i,
                    "drawn_colour": CLASS_NAMES[b["cls"]],
                    "box_frac": [
                        round(b["box"][0] / w, 4),
                        round(b["box"][1] / h, 4),
                        round(b["box"][2] / w, 4),
                        round(b["box"][3] / h, 4),
                    ],
                }
            )
        cv2.imwrite(os.path.join(args.out, name + ".png"), out)
        with open(os.path.join(args.out, name + ".boxes.json"), "w") as fh:
            json.dump({"frame": name, "width": w, "height": h, "boxes": meta}, fh, indent=1)
    print(f"prepared {len(frame_names(args.frames))} frames -> {args.out}")


def human_verdict(pre_dir, truth_dir, name):
    """What a human's corrected labels imply for each prelabel box (keep/delete/recolor),
    plus the robots they had to add. Same matching rules as evaluate()."""
    img = cv2.imread(os.path.join(pre_dir, "images", name + ".jpg"))
    h, w = img.shape[:2]
    pre = read_labels(os.path.join(pre_dir, "labels", name + ".txt"), w, h)
    truth = read_labels(os.path.join(truth_dir, name + ".txt"), w, h)
    used, boxes = set(), []
    for i, b in enumerate(pre):
        best, bi = 0.0, None
        for j, t in enumerate(truth):
            if j in used:
                continue
            v = iou(b["box"], t["box"])
            if v > best:
                best, bi = v, j
        if bi is not None and best >= 0.3:
            used.add(bi)
            if truth[bi]["cls"] == b["cls"]:
                boxes.append({"id": i, "verdict": "keep"})
            else:
                boxes.append(
                    {"id": i, "verdict": "recolor", "colour": CLASS_NAMES[truth[bi]["cls"]]}
                )
        else:
            boxes.append({"id": i, "verdict": "delete"})
    missed = [
        {
            "colour": CLASS_NAMES[t["cls"]],
            "box": [
                round(t["box"][0] / w, 3),
                round(t["box"][1] / h, 3),
                round(t["box"][2] / w, 3),
                round(t["box"][3] / h, 3),
            ],
        }
        for t in truth
        if not any(iou(t["box"], p_["box"]) >= 0.3 for p_ in pre)
    ]
    return {"boxes": boxes, "missed": missed, "unsure": False}


def evaluate(args):
    """Compare verdicts to human labels. Returns a metrics dict (no printing)."""
    tp = fp = fn = col_ok = col_n = 0
    dec_ok = dec_n = 0
    miss_true = miss_found = 0
    missing_files = 0
    per_verdict = {"keep": [0, 0], "delete": [0, 0], "recolor": [0, 0]}
    for name in frame_names(args.frames):
        vpath = os.path.join(args.verdicts, name + ".json")
        if not os.path.exists(vpath):
            missing_files += 1
            continue
        verdict = json.load(open(vpath))
        img = cv2.imread(os.path.join(args.pre, "images", name + ".jpg"))
        h, w = img.shape[:2]
        pre = read_labels(os.path.join(args.pre, "labels", name + ".txt"), w, h)
        truth = read_labels(os.path.join(args.truth, name + ".txt"), w, h)
        by_id = {v["id"]: v for v in verdict.get("boxes", [])}

        # what a human implicitly decided about each prelabel box
        final = []
        used = set()
        for i, b in enumerate(pre):
            best, bi = 0.0, None
            for j, t in enumerate(truth):
                if j in used:
                    continue
                v = iou(b["box"], t["box"])
                if v > best:
                    best, bi = v, j
            # 0.3 not 0.5: a loose prelabel the human redrew tighter is still a robot
            # the human kept; only boxes with no real overlap count as deleted.
            if bi is not None and best >= 0.3:
                used.add(bi)
                human = "keep" if truth[bi]["cls"] == b["cls"] else "recolor"
            else:
                human = "delete"
            v = by_id.get(i, {"verdict": "keep"})
            model = v.get("verdict", "keep")
            dec_n += 1
            dec_ok += model == human
            per_verdict[human][1] += 1
            per_verdict[human][0] += model == human
            if model == "delete":
                continue
            cls = b["cls"]
            if model == "recolor":
                cls = 1 - cls if v.get("colour") is None else (0 if v["colour"] == "blue" else 1)
            final.append({"cls": cls, "box": b["box"]})

        # robots the human had to ADD (no prelabel box overlaps them)
        added = [t for t in truth if not any(iou(t["box"], p["box"]) >= 0.3 for p in pre)]
        miss_true += len(added)
        for t in added:
            for m in verdict.get("missed", []):
                mb = [m["box"][0] * w, m["box"][1] * h, m["box"][2] * w, m["box"][3] * h]
                if iou(mb, t["box"]) >= 0.2:
                    miss_found += 1
                    break

        # end state vs truth
        used = set()
        for f in final:
            best, bj = 0.0, None
            for j, t in enumerate(truth):
                if j in used:
                    continue
                v = iou(f["box"], t["box"])
                if v > best:
                    best, bj = v, j
            if bj is not None and best >= 0.5:
                used.add(bj)
                tp += 1
                col_n += 1
                col_ok += f["cls"] == truth[bj]["cls"]
            else:
                fp += 1
        fn += len(truth) - len(used)

    return {
        "missing_files": missing_files,
        "dec_ok": dec_ok,
        "dec_n": dec_n,
        "per_verdict": per_verdict,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "col_ok": col_ok,
        "col_n": col_n,
        "miss_true": miss_true,
        "miss_found": miss_found,
    }


def score(args):
    m = evaluate(args)
    n = m["dec_n"] or 1
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    print(f"verdicts: {args.verdicts}  (missing files: {m['missing_files']})")
    print(f"per-box decision agreement: {m['dec_ok']}/{m['dec_n']} = {m['dec_ok'] / n:.1%}")
    for k, (ok, tot) in m["per_verdict"].items():
        print(f"   human said {k:8s}: model agreed {ok}/{tot}")
    print(f"after applying verdicts (no human-added boxes): TP={tp} FP={fp} FN={fn}")
    print(f"   precision {tp / max(1, tp + fp):.1%}   recall {tp / max(1, tp + fn):.1%}")
    print(f"   colour correct on matched boxes: {m['col_ok']}/{m['col_n']}")
    print(f"robots the human had to add: {m['miss_true']}; model pointed at {m['miss_found']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--pre", required=True, help="prelabel dir with images/ and labels/")
    a.add_argument("--frames", required=True, help="text file of frame names")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=prepare)
    b = sub.add_parser("score")
    b.add_argument("--pre", required=True)
    b.add_argument("--truth", required=True, help="human-corrected labels dir")
    b.add_argument("--verdicts", required=True)
    b.add_argument("--frames", required=True)
    b.set_defaults(fn=score)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
