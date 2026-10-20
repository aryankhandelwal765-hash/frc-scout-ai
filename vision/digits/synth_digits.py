"""
Synthetic bumper-number crops: random team numbers rendered like FRC bumpers.

Real crops carry only ~100 distinct numbers, so a reader trained on them memorises
robots instead of learning digit shapes (0% on unseen numbers). Rendering numbers we
choose gives it every digit in every position.

    python synth_digits.py --out ../digit_data_synth --n 40000

Output matches build_digit_crops.py: <out>/crops/*.jpg + <out>/labels.csv. Every row
has box_w 200 and match "synth".
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/System/Library/Fonts/Supplemental/"
FONTS = [
    "Arial Bold.ttf",
    "Arial Black.ttf",
    "Arial Narrow Bold.ttf",
    "Arial Rounded Bold.ttf",
    "Impact.ttf",
    "DIN Alternate Bold.ttf",
    "Tahoma Bold.ttf",
    "Trebuchet MS Bold.ttf",
    "Arial Bold Italic.ttf",
]
W, H = 240, 70


def rand_team(rng):
    r = rng.random()
    if r < 0.12:
        return str(rng.randint(1, 999))
    if r < 0.85:
        return str(rng.randint(1000, 9999))
    return str(rng.randint(10000, 11999))


def background(rng, alliance):
    if alliance == "red":
        base = np.array([rng.randint(150, 235), rng.randint(15, 70), rng.randint(15, 70)])
    else:
        base = np.array([rng.randint(10, 60), rng.randint(50, 130), rng.randint(150, 235)])
    img = np.zeros((H, W, 3), np.float32)
    img[:] = base
    img *= np.linspace(rng.uniform(0.7, 1.1), rng.uniform(0.6, 1.0), H)[:, None, None]
    top = rng.randint(0, 22)  # robot body above the bumper
    if top and rng.random() < 0.7:
        img[:top] = np.array([rng.randint(20, 120)] * 3) + np.random.randn(top, W, 3) * 12
    if rng.random() < 0.5:  # carpet below
        bot = rng.randint(3, 12)
        img[H - bot :] = np.array([rng.randint(70, 150)] * 3)
    return img


def render(rng, team):
    alliance = rng.choice(["red", "blue"])
    img = background(rng, alliance)
    font = ImageFont.truetype(FONT_DIR + rng.choice(FONTS), rng.randint(28, 56))
    layer = Image.new("L", (W * 2, H * 2), 0)
    d = ImageDraw.Draw(layer)
    bb = d.textbbox((0, 0), team, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    # fit the number into 38-92% of the width and 40-85% of the height
    sc = min(W * rng.uniform(0.38, 0.92) / max(tw, 1), H * rng.uniform(0.4, 0.85) / max(th, 1))
    d.text((W - tw // 2 - bb[0], H - th // 2 - bb[1]), team, font=font, fill=255)
    layer = layer.resize((int(W * 2 * sc), int(H * 2 * sc)), Image.LANCZOS)
    lw, lh = layer.size
    m = np.array(layer, np.float32) / 255.0
    cx = rng.randint(lw // 2 - 10, W - lw // 2 + 10) if lw < W else W // 2
    cy = rng.randint(max(th // 2, 25), H - 12) if True else H // 2
    ys, xs = cy - lh // 2, cx - lw // 2
    full = np.zeros((H, W), np.float32)
    y0, x0 = max(0, ys), max(0, xs)
    y1, x1 = min(H, ys + lh), min(W, xs + lw)
    if y1 > y0 and x1 > x0:
        full[y0:y1, x0:x1] = m[y0 - ys : y1 - ys, x0 - xs : x1 - xs]
    ink = np.array([rng.randint(215, 255)] * 3, np.float32)
    if rng.random() < 0.08:
        ink = np.array([255, 235, 150], np.float32)
    shadow = cv2.GaussianBlur(np.roll(full, 2, 0), (5, 5), 0)
    img = img * (1 - 0.35 * shadow[..., None])
    img = img * (1 - full[..., None]) + ink * full[..., None]
    # yellow balls (they pile up in front of the bumpers)
    for _ in range(rng.choice([0, 0, 0, 1, 1, 2, 3])):
        r = rng.randint(8, 18)
        cv2.circle(
            img, (rng.randint(0, W), rng.randint(H // 2, H)), r, (235, 215, rng.randint(20, 80)), -1
        )
    # shear (italic / viewing angle) + perspective + blur + noise
    sh = rng.uniform(-0.25, 0.25)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    j = lambda s: rng.uniform(-s, s)
    dst = src + np.float32(
        [[j(8) + sh * 20, j(5)], [j(8) - sh * 20, j(5)], [j(8), j(5)], [j(8), j(5)]]
    )
    img = cv2.warpPerspective(
        img, cv2.getPerspectiveTransform(src, dst), (W, H), borderMode=cv2.BORDER_REPLICATE
    )
    k = rng.choice([1, 3, 3, 5, 7])
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    f = rng.uniform(0.4, 0.9)  # downscale-upscale: video compression softness
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    img = (
        img * rng.uniform(0.6, 1.3)
        + rng.uniform(-25, 25)
        + np.random.randn(H, W, 3) * rng.uniform(0, 7)
    )
    return np.clip(img[..., ::-1], 0, 255).astype(np.uint8), alliance  # RGB -> BGR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)
    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "team", "match", "alliance", "box_w", "box_h"])
        for i in range(args.n):
            team = rand_team(rng)
            im, al = render(rng, team)
            name = f"synth_{i:06d}.jpg"
            cv2.imwrite(os.path.join(args.out, "crops", name), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            w.writerow([name, team, "synth", al, 200, 70])


if __name__ == "__main__":
    main()
