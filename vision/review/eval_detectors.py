"""
Score detectors on the same human-corrected frames, side by side.

    python eval_detectors.py --data dataset_v3/data.yaml \
        --weights old=runs/detect/frc_robots-2/weights/best.pt \
        --weights new=runs/experiments/frc_robots_v3/weights/best.pt --imgsz 960 1280

Uses the val split of data.yaml (frames a human corrected; never trained on). Reports
precision, recall, mAP@50 and mAP@50-95, and writes <data dir>/eval.json + eval.md.
"""

import argparse
import json
import os

DEVICE_PREF = ("mps", "cuda", "cpu")


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--weights", action="append", required=True, help="label=path (repeatable)")
    p.add_argument("--imgsz", type=int, nargs="+", default=[960, 1280])
    args = p.parse_args()

    from ultralytics import YOLO

    device = pick_device()
    rows = []
    for spec in args.weights:
        label, path = spec.split("=", 1)
        model = YOLO(path)
        for sz in args.imgsz:
            r = model.val(
                data=args.data,
                imgsz=sz,
                batch=4,
                device=device,
                conf=0.001,
                plots=False,
                verbose=False,
            )
            b = r.box
            rows.append(
                {
                    "model": label,
                    "imgsz": sz,
                    "precision": round(float(b.mp), 3),
                    "recall": round(float(b.mr), 3),
                    "map50": round(float(b.map50), 3),
                    "map50_95": round(float(b.map), 3),
                }
            )
            print(rows[-1], flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.data))
    json.dump(rows, open(os.path.join(out_dir, "eval.json"), "w"), indent=1)
    lines = [
        "| model | imgsz | precision | recall | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['model']} | {r['imgsz']} | {r['precision']} | {r['recall']} | {r['map50']} | "
        f"{r['map50_95']} |"
        for r in rows
    ]
    open(os.path.join(out_dir, "eval.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
