"""
Interactive mask lab -- run your own masks on the footage, one frame at a time.

Drag the HSV sliders to build a color mask; step through frames to see how it
holds up; press 's' to print the current values (copy them into config.py or your
own script). Optionally drop your own logic into custom_mask() below.

    python tune_color_masks.py --input test_clip.mp4                 # video, step frames
    python tune_color_masks.py --images dataset/images               # step the extracted frames
    python tune_color_masks.py --input test_clip.mp4 --crop 0 100 1280 470   # top field only

Keys:
    n / SPACE  next frame        p  previous frame
    ] / [      jump +/-30 frames
    s          print current HSV values + blob count
    m          toggle: HSV-slider mask  <->  custom_mask() below
    q / ESC    quit

Run from a Terminal (opens a window), not the PyCharm run console.
"""

import argparse
import glob
import os

import cv2
import numpy as np


def custom_mask(frame_bgr, hsv):
    """<-- EDIT THIS to run your own mask. Return a uint8 mask (0/255).
    Example: moving-ball / bumper / whatever you want to try. Toggle to it with 'm'."""
    # example: yellow balls, tight
    return cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))


def _fit(img, width=960):
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)))


def main():
    p = argparse.ArgumentParser(description="Interactive HSV / custom mask explorer.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="video file")
    src.add_argument("--images", help="directory of images to step through")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--crop", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--min-area", type=int, default=30, help="blob area for the count readout")
    args = p.parse_args()

    # frame source
    if args.images:
        paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                       glob.glob(os.path.join(args.images, "*.png")))
        if not paths:
            raise SystemExit(f"No images in {args.images}")
        n_frames = len(paths)
        get = lambda i: cv2.imread(paths[i])
    else:
        cap = cv2.VideoCapture(args.input)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        def get(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            return fr if ok else None

    win = "mask lab"
    cv2.namedWindow(win)
    for name, val in (("H lo", 18), ("H hi", 45), ("S lo", 55), ("S hi", 255),
                      ("V lo", 80), ("V hi", 255), ("open", 0), ("close", 2)):
        cv2.createTrackbar(name, win, val, 255, lambda _v: None)

    def tb(name):
        return cv2.getTrackbarPos(name, win)

    idx = max(0, min(args.start_frame, n_frames - 1))
    use_custom = False

    while True:
        frame = get(idx)
        if frame is None:
            break
        if args.crop:
            x1, y1, x2, y2 = args.crop
            frame = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if use_custom:
            mask = custom_mask(frame, hsv)
            label = "custom_mask()"
        else:
            lo = np.array([tb("H lo"), tb("S lo"), tb("V lo")])
            hi = np.array([tb("H hi"), tb("S hi"), tb("V hi")])
            mask = cv2.inRange(hsv, lo, hi)
            k_open, k_close = tb("open"), tb("close")
            if k_open > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
            if k_close > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
            label = "HSV sliders"

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in cnts if cv2.contourArea(c) >= args.min_area]

        overlay = frame.copy()
        overlay[mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)
        cv2.putText(overlay, f"frame {idx}/{n_frames-1}  {label}  blobs>={args.min_area}: {len(blobs)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        stack = np.vstack([_fit(overlay), _fit(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))])
        cv2.imshow(win, stack)

        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k in (ord("n"), ord(" ")):
            idx = min(n_frames - 1, idx + 1)
        elif k == ord("p"):
            idx = max(0, idx - 1)
        elif k == ord("]"):
            idx = min(n_frames - 1, idx + 30)
        elif k == ord("["):
            idx = max(0, idx - 30)
        elif k == ord("m"):
            use_custom = not use_custom
        elif k == ord("s"):
            if use_custom:
                print(f"frame {idx}: custom_mask(), blobs>={args.min_area}: {len(blobs)}")
            else:
                print(f"frame {idx}: HSV lo=({tb('H lo')},{tb('S lo')},{tb('V lo')}) "
                      f"hi=({tb('H hi')},{tb('S hi')},{tb('V hi')}) "
                      f"open={tb('open')} close={tb('close')} "
                      f"blobs>={args.min_area}: {len(blobs)}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
