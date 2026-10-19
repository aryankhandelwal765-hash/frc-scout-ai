"""
Build a YOLO training set from reviewed prelabels, keeping only labels we can trust.

    python build_training_set.py --pre refresh --prelabel-labels dataset_v2/labels \
        --mlx bulk/verdicts_mlx --opus bulk/verdicts_opus --out dataset_v3

Split:
  val   = frames a HUMAN corrected (their saved labels differ from the prelabel). These
          are gold; the model never trains on them, so metrics on val are honest.
  train = the rest, but only when the reviewers' verdicts leave a COMPLETE label set:
            - Opus reviewed it  -> apply Opus verdicts (keep / recolor / delete)
            - else cheap model  -> only if it said "keep" for every box
          Frames are dropped from training when the reviewer was unsure (cheap model
          only; Opus flags nearly every crowded frame, so its "unsure" is ignored),
          listed a robot the prelabel missed (we cannot draw it, so the label set is incomplete), or
          fewer than --min-boxes boxes survive. A missing label on a visible robot
          teaches the detector that robots are background, which is worse than less data.

Known limit: none of the reviewers has found robots the detector missed, so this set
cleans precision and colour but does NOT teach recall beyond what prelabel already had.

Output: <out>/images|labels/{train,val}, data.yaml, manifest.json
"""

import argparse
import json
import os
import shutil


def apply_verdict(lines, verdict, ignore_unsure=False):
    """Prelabel YOLO lines + verdict -> (kept lines, complete?).

    ignore_unsure: a strong reviewer flags "unsure" on most crowded frames while its box
    verdicts stay reliable, so only a listed MISSED robot (incomplete label set) rejects."""
    by_id = {b["id"]: b for b in verdict.get("boxes", [])}
    out = []
    for i, ln in enumerate(lines):
        v = by_id.get(i, {"verdict": "keep"})
        if v.get("verdict") == "delete":
            continue
        parts = ln.split()
        if v.get("verdict") == "recolor" and v.get("colour") in ("blue", "red"):
            parts[0] = "0" if v["colour"] == "blue" else "1"
        out.append(" ".join(parts))
    complete = not verdict.get("missed") and (ignore_unsure or not verdict.get("unsure"))
    return out, complete


def read_lines(path):
    return [ln.strip() for ln in open(path) if ln.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pre", required=True, help="prelabel dir with images/ and labels/")
    p.add_argument("--prelabel-labels", help="labels dir the human edited (default <pre>/labels)")
    p.add_argument("--mlx", required=True, help="cheap-model verdicts dir")
    p.add_argument("--opus", help="strong-model verdicts dir (hard frames)")
    p.add_argument("--out", required=True)
    p.add_argument("--min-boxes", type=int, default=3)
    args = p.parse_args()

    edited_dir = args.prelabel_labels or os.path.join(args.pre, "labels")
    names = sorted(
        f[:-4] for f in os.listdir(os.path.join(args.pre, "labels")) if f.endswith(".txt")
    )
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    def emit(n, split, lines):
        shutil.copy2(
            os.path.join(args.pre, "images", n + ".jpg"),
            os.path.join(args.out, "images", split, n + ".jpg"),
        )
        with open(os.path.join(args.out, "labels", split, n + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))

    stats = {"val_human": 0, "train_opus": 0, "train_cheap_clean": 0}
    dropped = {"no_verdict": 0, "unsure_or_missed": 0, "cheap_had_changes": 0, "too_few_boxes": 0}
    counts = {"train_boxes": 0, "val_boxes": 0}
    for n in names:
        pre_lines = read_lines(os.path.join(args.pre, "labels", n + ".txt"))
        edited = read_lines(os.path.join(edited_dir, n + ".txt"))
        if edited != pre_lines:  # a human touched it
            emit(n, "val", edited)
            stats["val_human"] += 1
            counts["val_boxes"] += len(edited)
            continue
        op = os.path.join(args.opus, n + ".json") if args.opus else None
        cp = os.path.join(args.mlx, n + ".json")
        if op and os.path.exists(op):
            lines, complete = apply_verdict(pre_lines, json.load(open(op)), ignore_unsure=True)
            kind = "train_opus"
        elif os.path.exists(cp):
            v = json.load(open(cp))
            if any(b.get("verdict") != "keep" for b in v.get("boxes", [])):
                dropped["cheap_had_changes"] += 1
                continue
            lines, complete = apply_verdict(pre_lines, v)
            kind = "train_cheap_clean"
        else:
            dropped["no_verdict"] += 1
            continue
        if not complete:
            dropped["unsure_or_missed"] += 1
            continue
        if len(lines) < args.min_boxes:
            dropped["too_few_boxes"] += 1
            continue
        emit(n, "train", lines)
        stats[kind] += 1
        counts["train_boxes"] += len(lines)

    out = os.path.abspath(args.out)
    with open(os.path.join(out, "data.yaml"), "w") as fh:
        fh.write(
            f"path: {out}\ntrain: images/train\nval: images/val\n"
            "names:\n  0: blue_robot\n  1: red_robot\n"
        )
    manifest = {"included": stats, "dropped": dropped, **counts}
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
