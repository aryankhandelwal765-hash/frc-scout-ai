"""
Compare an MLX vision model with EasyOCR at reading bumper numbers.

    ../venv/bin/python eval_vlm.py prep  --data ../digit_data --out /tmp/vlm_sample.json --n 300
    ../.venv-vlm/bin/python eval_vlm.py vlm --sample /tmp/vlm_sample.json --out /tmp/vlm_reads.json
    ../venv/bin/python eval_vlm.py score --sample /tmp/vlm_sample.json --reads /tmp/vlm_reads.json

prep   samples labelled crops, runs EasyOCR on each, stores the match's six TBA teams
vlm    asks the model twice per crop: an open read, and a pick among the six candidates
score  same accounting as eval_digits.py: exact, accepted (snapped to a candidate), right-when-accepted
       (needs mlx-vlm, which lives in its own venv because it pins a different transformers)
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "match_pipeline"))

DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
OPEN_PROMPT = (
    "This is a crop of a robot bumper from an FRC match video. Read the white team number "
    "printed on the bumper. Reply with ONLY the digits, or UNKNOWN if you cannot read them."
)
PICK_PROMPT = (
    "This is a crop of a robot bumper from an FRC match video. The robot is one of these "
    "teams: {cands}. Which team number is printed on the bumper? Reply with ONLY that "
    "number, or UNKNOWN if you cannot tell."
)


def prep(a):
    import cv2
    import read_numbers as rn

    rows = [
        r
        for r in csv.DictReader(open(os.path.join(a.data, "labels.csv")))
        if int(r["box_w"]) >= a.min_box_w
    ]
    random.Random(1).shuffle(rows)
    cands_of, out = {}, []
    for r in rows:
        if len(out) >= a.n:
            break
        cands = cands_of.setdefault(r["match"], rn.match_candidates(r["match"]))
        if not cands or r["team"] not in cands:
            continue
        path = os.path.join(a.data, "crops", r["file"])
        big = cv2.resize(
            cv2.imread(path), None, fx=rn.UPSCALE, fy=rn.UPSCALE, interpolation=cv2.INTER_CUBIC
        )
        out.append(
            {
                "file": path,
                "team": r["team"],
                "cands": sorted(cands),
                "easyocr": [t for t, _c in rn._ocr_digits(big)],
            }
        )
    json.dump(out, open(a.out, "w"))
    print(f"{len(out)} crops over {len({os.path.basename(o['file'])[:12] for o in out})} matches")


def vlm(a):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    sample = json.load(open(a.sample))
    model, processor = load(a.model)
    config = load_config(a.model)

    def ask(text, path):
        try:
            prompt = apply_chat_template(processor, config, text, num_images=1, enable_thinking=False)
        except TypeError:
            prompt = apply_chat_template(processor, config, text, num_images=1)
        res = generate(
            model, processor, prompt, image=[path], max_tokens=16, temperature=0.0, verbose=False
        )
        return (res.text if hasattr(res, "text") else str(res)).strip()

    t0 = time.time()
    for i, s in enumerate(sample):
        s["open"] = ask(OPEN_PROMPT, s["file"])
        s["pick"] = ask(PICK_PROMPT.format(cands=", ".join(s["cands"])), s["file"])
        if i % 25 == 0:
            print(i, f"{time.time() - t0:.0f}s", s["team"], s["easyocr"], s["open"], s["pick"], flush=True)
    json.dump(sample, open(a.out, "w"))
    print(f"{len(sample)} crops, {(time.time() - t0) / len(sample):.2f}s per crop (two asks)")


def score(a):
    import read_numbers as rn

    def snap(reads, cands):
        sc = sorted(
            ((max((rn._similarity(x, c) for x in reads), default=0.0), c) for c in cands),
            reverse=True,
        )
        if sc[0][0] >= rn.MIN_SCORE and sc[0][0] - sc[1][0] >= rn.MIN_MARGIN:
            return sc[0][1]
        return None

    rows = json.load(open(a.reads))
    n = len(rows)
    digits = lambda t: re.sub(r"\D", "", t or "")  # noqa: E731
    readers = {
        "easyocr": lambda r: ("snap", r["easyocr"]),
        "vlm-open": lambda r: ("snap", [digits(r["open"])] if digits(r["open"]) else []),
        "vlm-pick": lambda r: ("direct", digits(r["pick"]) if digits(r["pick"]) in r["cands"] else None),
    }
    for k, f in readers.items():
        exact = acc = right = 0
        for r in rows:
            mode, v = f(r)
            got = snap(v, r["cands"]) if mode == "snap" and v else (v if mode == "direct" else None)
            exact += bool(v) and (v[0] if mode == "snap" else v) == r["team"]
            if got:
                acc += 1
                right += got == r["team"]
        print(
            f"{k:9s} exact {exact / n:.3f}  accepted {acc / n:.3f}  "
            f"right-when-accepted {right / max(acc, 1):.3f}  correct overall {right / n:.3f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--min-box-w", type=int, default=50)
    p = sub.add_parser("vlm")
    p.add_argument("--sample", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p = sub.add_parser("score")
    p.add_argument("--sample")
    p.add_argument("--reads", required=True)
    a = ap.parse_args()
    {"prep": prep, "vlm": vlm, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
