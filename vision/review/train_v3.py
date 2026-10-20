"""
Train the next robot detector on a reviewed dataset (build_training_set.py output).

    python train_v3.py --data dataset_v3/data.yaml --name frc_robots_v3
    python train_v3.py --data dataset_v3/data.yaml --name frc_robots_v3 --resume

Differences from the original train_detector.py recipe:
  - yolov8m (25.9M params) instead of nano, 1280px instead of 960 -- robots are small
    and the production frames are 1920 wide
  - output goes to runs/experiments/, NOT runs/detect/. batch_trajectories.py and
    prelabel.py pick the newest runs/detect/*/weights/best.pt, so keeping experiments
    out of that glob means an unreviewed model can never silently become the one the
    pipeline uses. Promote it yourself (copy to runs/detect/<name>/weights/best.pt)
    once eval_detectors.py shows it beats the old one on the human-corrected frames.
  - bumper colour IS the class, so hue augmentation is turned down
"""

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.join(os.path.dirname(HERE), "runs", "experiments")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--name", default="frc_robots_v3")
    p.add_argument("--model", default=os.path.join(os.path.dirname(HERE), "yolov8m.pt"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--device", default=None, help="default: mps if available")
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    import torch
    from ultralytics import YOLO

    if os.path.abspath(args.project).rstrip("/").endswith(os.path.join("runs", "detect")):
        raise SystemExit("refusing to write into runs/detect (the pipeline's weight glob)")
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    if args.resume:
        last = os.path.join(args.project, args.name, "weights", "last.pt")
        YOLO(last).train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=False,
        workers=4,
        save_period=5,
        close_mosaic=args.close_mosaic,
        hsv_h=0.008,
        plots=False,
    )


if __name__ == "__main__":
    main()
