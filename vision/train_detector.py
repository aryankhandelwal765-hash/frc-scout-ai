"""
Step 3 of training a robot detector: train YOLO on your labeled frames.

Takes the flat dataset/{images,labels} produced by annotate.py, makes a
train/val split, writes data.yaml, and fine-tunes YOLOv8-nano. Uses Apple-Silicon
GPU (MPS) if available, else CPU.

    python train_detector.py --dataset dataset --epochs 120

Result: runs/detect/<name>/weights/best.pt  -- that's your robot detector.

Notes:
  - Only frames that have a label file (reviewed in annotate.py) are used. A
    reviewed frame with no boxes is a valid "background" example.
  - Small datasets train fine; ~120-200 frames with ~6 robots each is a start.
  - If MPS errors, add --device cpu (slower but reliable).
"""

import argparse
import glob
import os
import random
import shutil

import torch
from ultralytics import YOLO

CLASSES = ["blue_robot", "red_robot"]


def build_split(dataset, val_frac, seed):
    images = sorted(
        glob.glob(os.path.join(dataset, "images", "*.jpg"))
        + glob.glob(os.path.join(dataset, "images", "*.png"))
    )
    labels_dir = os.path.join(dataset, "labels")
    # keep only reviewed frames (label file exists)
    pairs = [
        (im, os.path.join(labels_dir, os.path.splitext(os.path.basename(im))[0] + ".txt"))
        for im in images
    ]
    pairs = [(im, lb) for im, lb in pairs if os.path.exists(lb)]
    if not pairs:
        raise SystemExit("No labeled frames found. Run annotate.py first.")

    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    split = {"val": pairs[:n_val], "train": pairs[n_val:]}

    out = os.path.join(dataset, "_split")
    if os.path.exists(out):
        shutil.rmtree(out)
    for sub in ("train", "val"):
        os.makedirs(os.path.join(out, "images", sub), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", sub), exist_ok=True)
        for im, lb in split[sub]:
            shutil.copy(im, os.path.join(out, "images", sub, os.path.basename(im)))
            shutil.copy(lb, os.path.join(out, "labels", sub, os.path.basename(lb)))

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
    print(f"Split: {len(split['train'])} train / {len(split['val'])} val")
    return yaml_path


def main():
    p = argparse.ArgumentParser(description="Train a YOLO robot detector on labeled frames.")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--model", default="yolov8n.pt", help="base weights to fine-tune")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=960, help="train resolution (robots are small)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--device", help="mps | cpu (default: auto)")
    p.add_argument("--name", default="frc_robots")
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    yaml_path = build_split(args.dataset, args.val_frac, seed=0)

    print(f"Training {args.model} on {device} for {args.epochs} epochs @ {args.imgsz}px ...")
    model = YOLO(args.model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        patience=30,
    )
    print(f"\nDone. Detector weights: runs/detect/{args.name}/weights/best.pt")
    print("Next: run match_pipeline/batch_trajectories.py (it picks the newest runs/detect/*/weights/best.pt).")


if __name__ == "__main__":
    main()
