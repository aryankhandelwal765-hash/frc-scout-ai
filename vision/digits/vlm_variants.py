"""
Try cheap ways to make the MLX vision model read bumpers better, on one fixed sample.

    ../.venv-vlm/bin/python vlm_variants.py --sample S/hard_sample.json --data ../digit_data --out S/variants.json

Variants (all open reads, snapped to the match's six teams the way identify.py does):
  base      the prompt eval_vlm.py uses
  prompt2   describes the bumper (white digits on red/blue fabric, 3-5 digits)
  prep      contrast-stretched 2x crop
  fewshot   four labelled example crops before the question (teams not in the sample)
  fewshot+prep
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "match_pipeline"))

from eval_vlm import DEFAULT_MODEL, OPEN_PROMPT  # noqa: E402

PROMPT2 = (
    "This image is a crop of an FRC robot. The bumper is the red or blue fabric strip along the "
    "bottom, with a white team number of 3 to 5 digits printed on it. Read that number. "
    "Reply with ONLY the digits, or UNKNOWN if you cannot read all of them."
)
SHOT_Q = "What is the team number on the bumper? Reply with ONLY the digits."


def prep_image(path, out_dir):
    import cv2

    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(lab[..., 0])
    out = os.path.join(out_dir, os.path.basename(path))
    cv2.imwrite(out, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return out


def pick_shots(data, sample, k=4):
    used_teams = {s["team"] for s in sample}
    used_matches = {os.path.basename(s["file"]).rsplit("_f", 1)[0] for s in sample}
    rows = [
        r
        for r in csv.DictReader(open(os.path.join(data, "labels.csv")))
        if r["team"] not in used_teams and r["match"] not in used_matches and int(r["box_w"]) >= 120
    ]
    random.Random(3).shuffle(rows)
    seen, shots = set(), []
    for r in rows:
        if r["team"] not in seen:
            seen.add(r["team"])
            shots.append((os.path.join(data, "crops", r["file"]), r["team"]))
        if len(shots) == k:
            break
    return shots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()

    import read_numbers as rn
    from mlx_vlm import generate, load
    from mlx_vlm.utils import load_config

    sample = json.load(open(a.sample))[: a.n]
    model, processor = load(a.model)
    config = load_config(a.model)
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.out)), "prep_crops")
    os.makedirs(tmp, exist_ok=True)
    shots = pick_shots(a.data, sample)
    print("shots:", [t for _p, t in shots], flush=True)

    def chat(messages, images):
        from mlx_vlm.prompt_utils import apply_chat_template

        try:
            prompt = apply_chat_template(
                processor, config, messages, num_images=len(images), enable_thinking=False
            )
        except TypeError:
            prompt = apply_chat_template(processor, config, messages, num_images=len(images))
        res = generate(
            model, processor, prompt, image=images, max_tokens=16, temperature=0.0, verbose=False
        )
        return (res.text if hasattr(res, "text") else str(res)).strip()

    def ask(variant, path):
        prep = "prep" in variant
        img = prep_image(path, tmp) if prep else path
        text = PROMPT2 if variant.startswith("prompt2") else OPEN_PROMPT
        if variant.startswith("fewshot"):
            shot_imgs = [prep_image(p, tmp) if prep else p for p, _t in shots]
            msgs = []
            for _p, t in shots:
                msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SHOT_Q}]})
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": t}]})
            msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SHOT_Q}]})
            return chat(msgs, [*shot_imgs, img])
        return chat(text, [img])

    variants = a.only or ["base", "prompt2", "prep", "fewshot", "fewshot+prep"]
    results = {}
    for v in variants:
        t0 = time.time()
        reads = [ask(v, s["file"]) for s in sample]
        results[v] = reads
        n = len(sample)
        digits = lambda t: re.sub(r"\D", "", t or "")  # noqa: E731
        acc = right = 0
        for s, r in zip(sample, reads):
            d = digits(r)
            if not d:
                continue
            sc = sorted(((rn._similarity(d, c), c) for c in s["cands"]), reverse=True)
            if sc[0][0] >= rn.MIN_SCORE and sc[0][0] - sc[1][0] >= rn.MIN_MARGIN:
                acc += 1
                right += sc[0][1] == s["team"]
        print(
            f"{v:13s} accepted {acc / n:.3f}  right-when-accepted {right / max(acc, 1):.3f}  "
            f"correct overall {right / n:.3f}   {(time.time() - t0) / n:.2f}s/crop",
            flush=True,
        )
        json.dump(results, open(a.out, "w"))


if __name__ == "__main__":
    main()
