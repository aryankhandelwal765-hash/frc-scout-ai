"""
Train a small CRNN + CTC reader for FRC bumper numbers on build_digit_crops.py output.

    python train_digits.py --data ../digit_data --out ../runs/digits/v1

Why CTC: team numbers are 1-5 digits with no fixed layout, so the model reads a digit
STRING off the bumper band rather than classifying a fixed set.

Held out BY TEAM: ~15% of teams never appear in training, so val accuracy says how the
reader does on numbers it has never seen (the next event), not just new frames of
robots it has memorised.

Label noise: labels are per-track, so some crops are mislabelled or unreadable. After
--clean-after epochs the model's own confident disagreements with a label are dropped
from training and the run continues on the cleaned set.
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H, W = 32, 128
CHARS = "0123456789"
BLANK = 0  # CTC blank; digit d is class d + 1


def encode(s):
    return [CHARS.index(c) + 1 for c in s]


def load_crop(path):
    im = cv2.imread(path)
    return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)


def augment(im):
    im = im.astype(np.float32)
    im = im * random.uniform(0.6, 1.4) + random.uniform(-30, 30)
    if random.random() < 0.5:
        k = random.choice((3, 5))
        im = cv2.GaussianBlur(im, (k, k), 0)
    ang, sh = random.uniform(-4, 4), random.uniform(-0.15, 0.15)
    sc = random.uniform(0.9, 1.1)
    M = np.array(
        [
            [sc, sh, random.uniform(-4, 4)],
            [np.tan(np.radians(ang)) * 0.3, sc, random.uniform(-2, 2)],
        ],
        np.float32,
    )
    im = cv2.warpAffine(im, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    im += np.random.randn(*im.shape) * random.uniform(0, 6)
    return np.clip(im, 0, 255)


def to_tensor(im):
    return torch.from_numpy(((im.astype(np.float32) / 255.0) - 0.5) / 0.5).permute(2, 0, 1)


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o, pool):
            return [
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(True),
                nn.MaxPool2d(pool),
            ]

        self.cnn = nn.Sequential(
            *blk(3, 32, (2, 2)), *blk(32, 64, (2, 2)), *blk(64, 128, (2, 1)), *blk(128, 128, (2, 1))
        )  # 32x128 -> 2x32
        self.rnn = nn.LSTM(
            128 * 2, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(256, len(CHARS) + 1)

    def forward(self, x):
        f = self.cnn(x)  # B,C,2,32
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        out, _ = self.rnn(f)
        return self.fc(out)  # B,T,classes


def decode(logits):
    """greedy CTC -> [(string, confidence)]"""
    p = F.softmax(logits, -1)
    conf, idx = p.max(-1)
    out = []
    for i in range(idx.shape[0]):
        s, last, cs = [], 0, []
        for t in range(idx.shape[1]):
            k = int(idx[i, t])
            if k != last and k != BLANK:
                s.append(CHARS[k - 1])
            if k != BLANK:
                cs.append(float(conf[i, t]))
            last = k
        out.append(("".join(s), float(np.mean(cs)) if cs else 0.0))
    return out


def batches(items, imgs, bs, train):
    idx = list(range(len(items)))
    if train:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        sel = idx[i : i + bs]
        x = torch.stack([to_tensor(augment(imgs[j]) if train else imgs[j]) for j in sel])
        tg = [torch.tensor(encode(items[j]["team"])) for j in sel]
        yield sel, x, tg


def evaluate(model, items, imgs, dev, cand_by_match=None):
    model.eval()
    exact = ok_snap = n = 0
    with torch.no_grad():
        for sel, x, _ in batches(items, imgs, 128, False):
            for j, (s, c) in zip(sel, decode(model(x.to(dev)).cpu())):
                n += 1
                exact += s == items[j]["team"]
    return exact / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--clean-after", type=int, default=15)
    ap.add_argument("--val-team-frac", type=float, default=0.15)
    ap.add_argument("--min-box-w", type=int, default=50)
    ap.add_argument("--synth", help="synth_digits.py output dir, added to TRAIN only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    rows = [r for r in csv.DictReader(open(os.path.join(args.data, "labels.csv")))]
    rows = [r for r in rows if int(r["box_w"]) >= args.min_box_w]  # real crops only here
    teams = sorted({r["team"] for r in rows})
    rng = random.Random(args.seed)
    val_teams = set(rng.sample(teams, max(1, int(len(teams) * args.val_team_frac))))
    train = [r for r in rows if r["team"] not in val_teams]
    val = [r for r in rows if r["team"] in val_teams]
    synth = []
    if args.synth:
        synth = [
            dict(r, dir=args.synth)
            for r in csv.DictReader(open(os.path.join(args.synth, "labels.csv")))
        ]
        train += synth
    print(
        f"{len(rows)} real crops (+{len(synth)} synthetic), {len(teams)} teams -> train {len(train)}, val {len(val)} ({len(val_teams)} unseen teams)"
    )
    load = lambda rs: [
        load_crop(os.path.join(r.get("dir", args.data), "crops", r["file"])) for r in rs
    ]
    tr_img, va_img = load(train), load(val)

    model = CRNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 3e-3, total_steps=args.epochs * ((len(train) + 63) // 64) + 10
    )
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    os.makedirs(args.out, exist_ok=True)
    best = 0.0
    for ep in range(args.epochs):
        if ep == args.clean_after:
            model.eval()
            keep_i = []
            with torch.no_grad():
                for sel, x, _ in batches(train, tr_img, 128, False):
                    for j, (s, c) in zip(sel, decode(model(x.to(dev)).cpu())):
                        if not (s != train[j]["team"] and c > 0.8):
                            keep_i.append(j)
            dropped = len(train) - len(keep_i)
            train = [train[j] for j in keep_i]
            tr_img = [tr_img[j] for j in keep_i]
            print(f"  clean: dropped {dropped} confidently-disagreeing crops, {len(train)} left")
        model.train()
        tot = 0.0
        for sel, x, tg in batches(train, tr_img, 64, True):
            logits = model(x.to(dev)).log_softmax(-1).permute(1, 0, 2).cpu()  # T,B,C
            lens_in = torch.full((len(sel),), logits.shape[0], dtype=torch.long)
            lens_tg = torch.tensor([len(t) for t in tg])
            loss = ctc(logits, torch.cat(tg), lens_in, lens_tg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss)
        acc = evaluate(model, val, va_img, dev)
        print(
            f"ep {ep + 1:2d} loss {tot / max(1, (len(train) + 63) // 64):.3f} val exact (unseen teams) {acc:.3f}",
            flush=True,
        )
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), os.path.join(args.out, "digits.pt"))
    print(f"best val exact {best:.3f}")
    with open(os.path.join(args.out, "val_teams.txt"), "w") as fh:
        fh.write("\n".join(sorted(val_teams)))


if __name__ == "__main__":
    main()
