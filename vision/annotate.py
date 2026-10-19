"""
Step 2 of training a robot detector: label robots by bumper color.

A self-contained OpenCV annotator -- no account, no web tool. For each frame,
drag a box around each robot and tag it its alliance (bumper color). Writes
YOLO-format labels next to the images (dataset/labels/*.txt).

    python annotate.py --images dataset/images

Controls (shown on screen too):
    left-drag   draw a box around a robot
    b / r       set current label to BLUE / RED (drawn boxes use this)
    t           toggle the label of the last box
    u           undo last box        c   clear this frame
    n / SPACE   save + next frame     p   save + previous frame
    q / ESC     save + quit

Classes: 0 = blue_robot, 1 = red_robot.
Run this from a Terminal (it opens a window) -- not the PyCharm run console.
"""

import argparse
import glob
import os

import cv2

CLASSES = ["blue_robot", "red_robot"]
COLORS = [(255, 120, 0), (0, 80, 255)]  # blue, red (BGR)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Annotator:
    def __init__(self, image_paths, labels_dir):
        self.paths = image_paths
        self.labels_dir = labels_dir
        self.idx = 0
        self.cls = 0  # current label (0=blue)
        self.boxes = []  # [(x1,y1,x2,y2,cls)]
        self.drawing = False
        self.p0 = None
        self.cur = None
        self.img = None
        os.makedirs(labels_dir, exist_ok=True)

    def label_path(self, image_path):
        base = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.labels_dir, base + ".txt")

    def load(self):
        self.img = cv2.imread(self.paths[self.idx])
        self.boxes = []
        h, w = self.img.shape[:2]
        lp = self.label_path(self.paths[self.idx])
        if os.path.exists(lp):
            for line in open(lp):
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = parts
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                self.boxes.append(
                    (int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2), int(c))
                )

    def save(self):
        h, w = self.img.shape[:2]
        with open(self.label_path(self.paths[self.idx]), "w") as f:
            for x1, y1, x2, y2, c in self.boxes:
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
                if bw < 0.005 or bh < 0.005:
                    continue
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.p0 = (x, y)
            self.cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x0, y0 = self.p0
            if abs(x - x0) > 3 and abs(y - y0) > 3:
                self.boxes.append((x0, y0, x, y, self.cls))
            self.p0 = self.cur = None

    def render(self):
        vis = self.img.copy()
        for x1, y1, x2, y2, c in self.boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLORS[c], 2)
            cv2.putText(
                vis, CLASSES[c][0].upper(), (min(x1, x2), min(y1, y2) - 4), FONT, 0.5, COLORS[c], 2
            )
        if self.drawing and self.p0 and self.cur:
            cv2.rectangle(vis, self.p0, self.cur, COLORS[self.cls], 1)
        nblue = sum(1 for *_, c in self.boxes if c == 0)
        nred = sum(1 for *_, c in self.boxes if c == 1)
        hud = (
            f"[{self.idx + 1}/{len(self.paths)}] {os.path.basename(self.paths[self.idx])}"
            f"  draw={CLASSES[self.cls]}  blue={nblue} red={nred}"
        )
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(vis, hud, (8, 18), FONT, 0.5, (255, 255, 255), 1)
        cv2.putText(
            vis,
            "b/r label  t toggle  u undo  c clear  n/SPACE next  p prev  q quit",
            (8, vis.shape[0] - 8),
            FONT,
            0.45,
            (0, 255, 255),
            1,
        )
        return vis

    def run(self):
        win = "annotate robots (b=blue r=red, drag boxes)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.on_mouse)
        self.load()
        while True:
            cv2.imshow(win, self.render())
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                self.save()
                break
            elif k == ord("b"):
                self.cls = 0
            elif k == ord("r"):
                self.cls = 1
            elif k == ord("t") and self.boxes:
                x1, y1, x2, y2, c = self.boxes[-1]
                self.boxes[-1] = (x1, y1, x2, y2, 1 - c)
            elif k == ord("u") and self.boxes:
                self.boxes.pop()
            elif k == ord("c"):
                self.boxes = []
            elif k in (ord("n"), ord(" ")):
                self.save()
                if self.idx < len(self.paths) - 1:
                    self.idx += 1
                    self.load()
            elif k == ord("p"):
                self.save()
                if self.idx > 0:
                    self.idx -= 1
                    self.load()
        cv2.destroyAllWindows()
        labeled = len(glob.glob(os.path.join(self.labels_dir, "*.txt")))
        print(f"Saved labels for {labeled} frames in {self.labels_dir}/")
        print("Next: python train_detector.py --dataset", os.path.dirname(self.labels_dir))


def main():
    p = argparse.ArgumentParser(description="Annotate robots by bumper color for YOLO training.")
    p.add_argument("--images", default="dataset/images")
    p.add_argument("--labels", help="default: sibling 'labels' dir")
    p.add_argument("--order", help="text file of image paths (e.g. prelabel's review_order.txt)")
    args = p.parse_args()
    if args.order:
        paths = [ln.strip() for ln in open(args.order) if ln.strip()]
    else:
        paths = sorted(
            glob.glob(os.path.join(args.images, "*.jpg"))
            + glob.glob(os.path.join(args.images, "*.png"))
        )
    if not paths:
        raise SystemExit(f"No images in {args.images} -- run extract_frames.py first.")
    labels = args.labels or os.path.join(os.path.dirname(args.images.rstrip("/")), "labels")
    Annotator(paths, labels).run()


if __name__ == "__main__":
    main()
