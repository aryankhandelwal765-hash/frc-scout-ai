"""
Pick the frames a cheap reviewer should NOT be trusted on, so a stronger model (Opus)
reviews only those. Weighted toward COLOUR doubts, since the cheap model almost never
recolours (it caught 1 of 4 on the hand-checked set).

    python route_hard.py --verdicts bulk/verdicts_mlx --inputs bulk/in \
        --prelabel-review refresh/review.json --out bulk/hard --batches 8 --cap 200

Signals (per frame, summed):
  recolor verdict                       +4 each   the model itself doubts the colour
  colour mismatch in its own "why"      +4 each   says "blue robot" about a red box, yet "keep"
  prelabel had no bumper colour         +2        colour was a guess (defaulted/eliminated)
  model unsure                          +3
  model lists a missed robot            +3
  detectors disagreed / low conf        +1
  box count far from 6                  +1

Writes <out>/hard_frames.txt (ranked), <out>/batch_<k>.txt, and <out>/scores.json.
"""

import argparse
import json
import os
import re


def why_colour(why):
    """First alliance colour a reason text commits to, or None."""
    m = re.search(r"\b(blue|red)\b", (why or "").lower())
    return m.group(1) if m else None


def score_frame(verdict, meta, pre_review):
    drawn = {b["id"]: b["drawn_colour"] for b in meta["boxes"]}
    s, why = 0, []
    for b in verdict.get("boxes", []):
        if b.get("verdict") == "recolor":
            s += 4
            why.append(f"recolor box {b['id']}")
        elif b.get("verdict") == "keep":
            c = why_colour(b.get("why"))
            if c and drawn.get(b.get("id")) and c != drawn[b["id"]]:
                s += 4
                why.append(f"why says {c}, drawn {drawn[b['id']]} (box {b['id']})")
    if verdict.get("unsure"):
        s += 3
        why.append("unsure")
    if verdict.get("missed"):
        s += 3
        why.append("missed robot")
    reasons = (pre_review or {}).get("reasons", [])
    if "no_bumper_colour" in reasons:
        s += 2
        why.append("prelabel colour guessed")
    if {"detectors_disagree", "low_conf"} & set(reasons):
        s += 1
    n = len(meta["boxes"])
    if n < 4 or n > 8:
        s += 1
    return s, why


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--verdicts", required=True)
    p.add_argument("--inputs", required=True, help="dir with N.boxes.json")
    p.add_argument("--prelabel-review", help="review.json from prelabel.py")
    p.add_argument("--out", required=True)
    p.add_argument("--min-score", type=int, default=3)
    p.add_argument("--cap", type=int, default=200)
    p.add_argument("--batches", type=int, default=8)
    args = p.parse_args()

    review = json.load(open(args.prelabel_review)) if args.prelabel_review else {}
    rows = []
    for f in sorted(os.listdir(args.verdicts)):
        if not f.endswith(".json") or f == "stats.json":
            continue
        n = f[:-5]
        verdict = json.load(open(os.path.join(args.verdicts, f)))
        meta = json.load(open(os.path.join(args.inputs, n + ".boxes.json")))
        s, why = score_frame(verdict, meta, review.get(n))
        rows.append((s, n, why))
    rows.sort(key=lambda r: (-r[0], r[1]))
    hard = [r for r in rows if r[0] >= args.min_score][: args.cap]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "hard_frames.txt"), "w") as fh:
        fh.write("\n".join(r[1] for r in hard) + ("\n" if hard else ""))
    for k in range(args.batches):
        with open(os.path.join(args.out, f"batch_{k}.txt"), "w") as fh:
            part = [r[1] for r in hard[k :: args.batches]]
            fh.write("\n".join(part) + ("\n" if part else ""))
    json.dump(
        {r[1]: {"score": r[0], "why": r[2]} for r in rows},
        open(os.path.join(args.out, "scores.json"), "w"),
        indent=1,
    )
    print(f"{len(rows)} reviewed, {len(hard)} routed to the stronger model (cap {args.cap})")


if __name__ == "__main__":
    main()
