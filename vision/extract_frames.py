"""
Step 1 of training a robot detector: pull a diverse set of frames to annotate.

Samples frames evenly across the match and saves the (cropped) top-field region,
so the detector is trained on exactly the view it will run on. Skips near-duplicate
frames so your labeling effort isn't wasted on identical scenes.

    python extract_frames.py --input test_clip.mp4 --n 150 --outdir dataset/images

Then label them with annotate.py.
"""

import argparse
import os

import cv2
import numpy as np

DEFAULT_CROP = (0, 100, 1280, 470)  # top overhead field for this feed


def main():
    p = argparse.ArgumentParser(description="Extract frames to annotate for robot detection.")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", default="dataset/images")
    p.add_argument("--n", type=int, default=150, help="how many frames to keep")
    p.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=DEFAULT_CROP,
        help="region to keep (match the analysis crop)",
    )
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, help="default: last readable scoreboard frame")
    p.add_argument(
        "--dup-thresh",
        type=float,
        default=6.0,
        help="skip a frame if mean pixel diff from the last kept one is below this",
    )
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(args.input)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = args.end_frame or total
    x1, y1, x2, y2 = args.crop

    # oversample, then thin out near-duplicates down to ~n
    step = max(1, (end - args.start_frame) // (args.n * 3))
    kept = 0
    last_small = None
    f = args.start_frame
    while f < end and kept < args.n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y1:y2, x1:x2]
        small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 36))
        if last_small is None or np.mean(cv2.absdiff(small, last_small)) >= args.dup_thresh:
            cv2.imwrite(os.path.join(args.outdir, f"frame_{f:06d}.jpg"), crop)
            last_small = small
            kept += 1
        f += step
    cap.release()
    print(f"Saved {kept} frames to {args.outdir}/  (crop {args.crop})")
    print("Next: python annotate.py --images", args.outdir)


if __name__ == "__main__":
    main()
