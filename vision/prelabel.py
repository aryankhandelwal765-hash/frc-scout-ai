"""
Pre-label training frames so annotation becomes review-and-correct, not draw-from-scratch.

Samples frames across many match videos (the SAME crop the production detector sees),
runs two detectors, merges their boxes, assigns alliance from bumper colour, and writes
YOLO labels plus a review queue ordered by how likely each frame is to need a human.

    python prelabel.py --videos-dir match_pipeline/videos --matches 40 --per-video 15 \
        --out dataset_v2

Detectors (union, NMS-merged). Each is a "vote":
  ft@<scale>  the project's own best.pt, run at each --scales size (default 960,1280)
  world       OPTIONAL open-vocabulary YOLO-World via --world PATH. Measured useless
              here: it finds people on this footage but returns zero "robot" boxes
              (tried several prompts, CPU and MPS, conf down to 0.03), so it is off.

A box found by every vote is high trust; found by fewer is flagged. Boxes on static field
structure (hub towers, ramps) are dropped using the same empty-field trick as
trajectories.py. Alliance comes from bumper colour (trajectories.alliance_from_masks),
not from a detector class, because the class is unreliable.

Output (dataset_v2/):
  images/<match>_f<frame>.jpg   cropped field frames
  labels/<match>_f<frame>.txt   YOLO labels (0 blue_robot, 1 red_robot)
  previews/*.jpg                boxes drawn, flagged ones in yellow
  review.json                   per-image flags + priority
  review_order.txt              image paths, most-needs-review first

Then correct them:  python annotate.py --images dataset_v2/images --order dataset_v2/review_order.txt
"""

import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

try:  # python.org framework Python ships no CA bundle; YOLO-World downloads CLIP on first use
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "match_pipeline"))
import trajectories as tj  # noqa: E402

EXPECTED_ROBOTS = 6
BLUE, RED = 0, 1
COL = {BLUE: (255, 120, 0), RED: (0, 80, 255)}
FLAG_COL = (0, 220, 255)


def pick_device(requested):
    return tj._pick_device(requested)


def list_videos(videos_dir):
    vids = sorted(
        p for p in glob.glob(os.path.join(videos_dir, "*.mp4")) if not p.endswith(".part")
    )
    return vids


def choose_videos(vids, n, seed):
    """Spread across the event: every playoff-style match is eligible equally with quals,
    but selection is stratified by name order so early/mid/late matches are all hit."""
    if n >= len(vids):
        return vids
    rng = random.Random(seed)
    step = len(vids) / float(n)
    return [vids[min(len(vids) - 1, int(i * step + rng.random() * step))] for i in range(n)]


def sample_frames(video, per_video, seed):
    """Jittered-uniform frame indices inside the played part of the match, deduped by look."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    lo, hi = int(total * 0.06), int(total * 0.94)
    rng = random.Random(f"{seed}:{video}")
    # oversample 2x then thin near-duplicates, same idea as extract_frames.py
    cand = sorted(rng.sample(range(lo, hi), min(per_video * 2, hi - lo)))
    out, last = [], None
    for f in cand:
        if len(out) >= per_video:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
        if last is None or float(np.mean(cv2.absdiff(small, last))) >= 6.0:
            out.append((f, frame))
            last = small
    cap.release()
    return out


def detect(model, img, conf, imgsz, device):
    res = model.predict(
        img, conf=conf, imgsz=imgsz, agnostic_nms=True, verbose=False, device=device
    )[0]
    if res.boxes is None:
        return []
    return [(tuple(float(v) for v in b.xyxy[0]), float(b.conf[0])) for b in res.boxes]


def merge(votes, iou_thr=0.5):
    """Union of detector outputs. votes = [(name, [(box, conf)])].
    Each result: dict(box, conf, src={names that found it})."""
    merged = []
    for src, dets in votes:
        for box, c in dets:
            for m in merged:
                if tj._iou(box, m["box"]) >= iou_thr:
                    m["src"].add(src)
                    if c > m["conf"]:
                        m["conf"], m["box"] = c, box
                    break
            else:
                merged.append({"box": box, "conf": c, "src": {src}})
    return merged


def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    return f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--videos-dir", default=os.path.join(HERE, "match_pipeline", "videos"))
    p.add_argument("--matches", type=int, default=40, help="how many videos to sample from")
    p.add_argument("--per-video", type=int, default=15)
    p.add_argument("--out", default=os.path.join(HERE, "dataset_v2"))
    p.add_argument("--world", help="optional YOLO-World weights (see docstring: not useful)")
    p.add_argument("--world-prompt", default="robot")
    p.add_argument("--scales", default="960,1280", help="finetuned inference sizes, one vote each")
    p.add_argument("--finetuned", help="default: newest runs/detect/*/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.15, help="low on purpose; review prunes")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="*", help="video stems to use instead of sampling")
    args = p.parse_args()

    from ultralytics import YOLO

    device = pick_device(args.device)
    fine_path = args.finetuned
    if not fine_path:
        cands = sorted(
            glob.glob(os.path.join(HERE, "runs", "detect", "*", "weights", "best.pt")),
            key=os.path.getmtime,
        )
        fine_path = cands[-1] if cands else None
    world_model = None
    if args.world:
        world_model = YOLO(args.world)
        world_model.set_classes([args.world_prompt])
    if not fine_path:
        raise SystemExit("no finetuned weights found; pass --finetuned")
    fine_model = YOLO(fine_path)
    scales = [int(x) for x in args.scales.split(",")]
    print(f"device={device} finetuned={fine_path} scales={scales} world={args.world}")

    vids = list_videos(args.videos_dir)
    if args.only:
        vids = [v for v in vids if os.path.splitext(os.path.basename(v))[0] in set(args.only)]
    else:
        vids = choose_videos(vids, args.matches, args.seed)
    if not vids:
        raise SystemExit(f"no videos found in {args.videos_dir}")

    for d in ("images", "labels", "previews"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    review = {}
    for vi, video in enumerate(vids, 1):
        stem = os.path.splitext(os.path.basename(video))[0]
        frames = sample_frames(video, args.per_video, args.seed)
        if not frames:
            print(f"[{vi}/{len(vids)}] {stem}: unreadable, skipped")
            continue
        x1, y1, x2, y2 = tj.field_crop(frames[0][1].shape)
        crop_box = (x1, y1, x2, y2)

        # empty-field model of THIS video: static-structure colour masks + FP boxes
        med = tj.median_frame(video, crop_box)
        bg_b, bg_r = tj.static_color_masks_from(med) if med is not None else (None, None)
        fp = []
        for sz in scales:
            fp += tj.static_fp_boxes(fine_model, med, conf=args.conf, imgsz=sz, device=device)
        if world_model:
            fp += tj.static_fp_boxes(
                world_model, med, conf=args.conf, imgsz=args.imgsz, device=device
            )

        for f, frame in frames:
            crop = frame[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            votes = [(f"ft{sz}", detect(fine_model, crop, args.conf, sz, device)) for sz in scales]
            if world_model:
                votes.append(("world", detect(world_model, crop, args.conf, args.imgsz, device)))
            dets = merge(votes)
            n_votes = len(votes)
            dets = [d for d in dets if not tj.is_static_fp(d["box"], fp)]
            bm, rm, _ = tj.frame_bumper_masks(crop, bg_b, bg_r)

            lines, flags, preview = [], [], crop.copy()
            for d in dets:
                alliance = tj.alliance_from_masks(bm, rm, d["box"])
                all_agree = len(d["src"]) == n_votes
                reasons = []
                if alliance is None:
                    reasons.append("no_bumper_colour")
                    alliance = "blue"
                if not all_agree:
                    reasons.append("detectors_disagree")
                if d["conf"] < 0.3:
                    reasons.append("low_conf")
                cls = BLUE if alliance == "blue" else RED
                lines.append(yolo_line(cls, d["box"], w, h))
                bx = [int(v) for v in d["box"]]
                cv2.rectangle(preview, bx[:2], bx[2:], FLAG_COL if reasons else COL[cls], 2)
                cv2.putText(
                    preview,
                    f"{d['conf']:.2f}",
                    (bx[0], max(12, bx[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    FLAG_COL if reasons else COL[cls],
                    1,
                )
                flags.append(reasons)

            name = f"{stem}_f{f:06d}"
            cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), crop)
            with open(os.path.join(args.out, "labels", name + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            cv2.imwrite(os.path.join(args.out, "previews", name + ".jpg"), preview)

            n_flagged = sum(1 for r in flags if r)
            count_off = abs(len(dets) - EXPECTED_ROBOTS)
            review[name] = {
                "n_boxes": len(dets),
                "flagged": n_flagged,
                "count_off": count_off,
                "reasons": sorted({x for r in flags for x in r}),
                "priority": n_flagged * 2 + count_off,
            }
        print(f"[{vi}/{len(vids)}] {stem}: {len(frames)} frames")

    with open(os.path.join(args.out, "review.json"), "w") as fh:
        json.dump(review, fh, indent=1)
    order = sorted(review, key=lambda k: -review[k]["priority"])
    with open(os.path.join(args.out, "review_order.txt"), "w") as fh:
        for k in order:
            fh.write(os.path.join(args.out, "images", k + ".jpg") + "\n")

    tot = len(review)
    clean = sum(1 for v in review.values() if v["priority"] == 0)
    print(f"\n{tot} frames -> {args.out}  ({clean} look clean, {tot - clean} need review)")
    print("Review order: ", os.path.join(args.out, "review_order.txt"))


if __name__ == "__main__":
    main()
