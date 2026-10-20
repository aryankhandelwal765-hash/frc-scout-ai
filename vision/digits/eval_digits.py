"""
Compare the digit reader with EasyOCR on held-out-team crops.

    python eval_digits.py --data ../digit_data --run ../runs/digits/v2 [--n 400]

On crops of teams the model never trained on, for each reader:
  exact     read string == label
  accepted  share of crops where the read snaps to one of the match's six TBA teams
            under identify.py's rule (MIN_SCORE, MIN_MARGIN)
  right     share of accepted snaps that equal the label
"""

import argparse
import csv
import os
import random
import sys

import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "match_pipeline"))
import read_numbers as rn  # noqa: E402
import train_digits as td  # noqa: E402


def snap(reads, cands):
    sc = sorted(
        ((max((rn._similarity(x, c) for x in reads), default=0.0), c) for c in cands), reverse=True
    )
    if sc[0][0] >= rn.MIN_SCORE and sc[0][0] - sc[1][0] >= rn.MIN_MARGIN:
        return sc[0][1]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--min-box-w", type=int, default=50)
    a = ap.parse_args()
    val_teams = set(open(os.path.join(a.run, "val_teams.txt")).read().split())
    rows = [
        r
        for r in csv.DictReader(open(os.path.join(a.data, "labels.csv")))
        if r["team"] in val_teams and int(r["box_w"]) >= a.min_box_w
    ]
    random.Random(1).shuffle(rows)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = td.CRNN().to(dev)
    model.load_state_dict(torch.load(os.path.join(a.run, "digits.pt"), map_location=dev))
    model.eval()
    cands_of = {}
    st = {k: [0, 0, 0] for k in ("crnn", "easyocr", "both")}  # exact, accepted, accepted-and-right
    n = 0
    for r in rows:
        if n >= a.n:
            break
        cands = cands_of.setdefault(r["match"], rn.match_candidates(r["match"]))
        if not cands:
            continue
        n += 1
        path = os.path.join(a.data, "crops", r["file"])
        x = td.to_tensor(td.load_crop(path)).unsqueeze(0).to(dev)
        with torch.no_grad():
            s, _c = td.decode(model(x).cpu())[0]
        big = cv2.resize(
            cv2.imread(path), None, fx=rn.UPSCALE, fy=rn.UPSCALE, interpolation=cv2.INTER_CUBIC
        )
        eo = [t for t, _c in rn._ocr_digits(big)]
        for k, reads in (
            ("crnn", [s] if s else []),
            ("easyocr", eo),
            ("both", ([s] if s else []) + eo),
        ):
            st[k][0] += bool(reads) and reads[0] == r["team"]
            sn = snap(reads, cands) if reads else None
            if sn:
                st[k][1] += 1
                st[k][2] += sn == r["team"]
    print(f"{n} held-out crops")
    for k, (ex, acc, right) in st.items():
        print(
            f"{k:8s} exact {ex / n:.3f}  accepted {acc / n:.3f}  right-when-accepted {right / max(acc, 1):.3f}  correct overall {right / n:.3f}"
        )


if __name__ == "__main__":
    main()
