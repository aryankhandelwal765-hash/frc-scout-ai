"""
Turn the labelled bumper crops into an mlx_vlm fine-tuning set, held out by TEAM.

    python build_vlm_data.py --data ../digit_data --out ../vlm_data [--val-frac 0.15]

Writes <out>/train.jsonl, <out>/valid.jsonl and <out>/val_teams.txt. A held-out team never appears
in train, so the validation score measures reading, not remembering which team looks like what.
The prompt is eval_vlm.OPEN_PROMPT so the tuned model answers the question the eval asks.
"""

import argparse
import csv
import json
import os
import random

from eval_vlm import OPEN_PROMPT


def row(path, team):
    return {
        "images": [path],
        "messages": [
            {"role": "user", "content": OPEN_PROMPT},
            {"role": "assistant", "content": team},
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--min-box-w", type=int, default=80)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    rows = [
        r
        for r in csv.DictReader(open(os.path.join(a.data, "labels.csv")))
        if int(r["box_w"]) >= a.min_box_w
    ]
    teams = sorted({r["team"] for r in rows})
    rng = random.Random(a.seed)
    rng.shuffle(teams)
    val_teams = set(teams[: max(1, int(len(teams) * a.val_frac))])
    train = [r for r in rows if r["team"] not in val_teams]
    val = [r for r in rows if r["team"] in val_teams]
    rng.shuffle(train)
    rng.shuffle(val)

    # cap per team so a few busy teams do not dominate
    per_team, capped = {}, []
    for r in train:
        per_team[r["team"]] = per_team.get(r["team"], 0) + 1
        if per_team[r["team"]] <= max(8, a.max_train // max(len(teams) - len(val_teams), 1) * 2):
            capped.append(r)
    train = capped[: a.max_train]
    val = val[: a.max_val]

    os.makedirs(a.out, exist_ok=True)
    crops = os.path.abspath(os.path.join(a.data, "crops"))
    for name, rs in (("train", train), ("valid", val)):
        with open(os.path.join(a.out, name + ".jsonl"), "w") as fh:
            for r in rs:
                fh.write(json.dumps(row(os.path.join(crops, r["file"]), r["team"])) + "\n")
    open(os.path.join(a.out, "val_teams.txt"), "w").write("\n".join(sorted(val_teams)))
    print(
        f"teams {len(teams)} (val {len(val_teams)}); "
        f"train {len(train)} crops; valid {len(val)} crops"
    )


if __name__ == "__main__":
    main()
