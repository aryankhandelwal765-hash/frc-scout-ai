"""
Cross-check the vision estimate against TBA ground truth, per alliance, per match.

Vision gives a per-alliance "balls scored" count (scoreboard OCR). TBA gives the
official score_breakdown. This joins them on match key and reports agreement.

Because the 2026 score_breakdown schema is game-specific, the TBA field to compare
against is configurable. Point --tba-field at the per-alliance game-piece count in
the breakdown, e.g. --tba-field autoCellsScored (a dotted path is allowed for
nested fields). Discover field names first with:

  python compare_tba.py --show-breakdown 2026hop_qm1

Usage:
  python compare_tba.py --show-breakdown 2026hop_qm1   # print one match's breakdown
  python compare_tba.py --tba-field totalGamePieces    # run the comparison
  python compare_tba.py                                 # defaults to alliance score

Outputs report.csv + report.json + a printed summary (MAE, bias, within-N rate).
"""

import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TRUTH = os.path.join(HERE, "truth")

# ===========================================================================
# EDIT ME (PyCharm: click Run — no terminal args needed). CLI args override.
# Step 1: set SHOW_BREAKDOWN to a match key, Run, read the printed fields.
# Step 2: clear SHOW_BREAKDOWN, set TBA_FIELD to the game-piece field, Run.
# ===========================================================================
SHOW_BREAKDOWN = ""  # set to a match key to print its TBA breakdown + exit
TBA_FIELD = "hubScore.totalCount"  # AGE balls scored per alliance (verified on 2026hop_qm1)
WITHIN = 10  # count an estimate "correct" if within +/- this many balls
# ===========================================================================


def load_truth():
    out = {}
    for f in glob.glob(os.path.join(TRUTH, "*.json")):
        t = json.load(open(f))
        out[t["key"]] = t
    return out


def dotted(d, path):
    """Fetch a nested value by dotted path; None if any hop is missing."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def tba_value(truth, alliance, field):
    """Ground-truth number for one alliance. field=None -> the alliance score;
    otherwise a (dotted) path into score_breakdown[alliance]."""
    if field is None:
        return truth.get(f"{alliance}_score")
    sb = truth.get("score_breakdown") or {}
    side = sb.get(alliance) or {}
    return dotted(side, field)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tba-field",
        default=TBA_FIELD,
        help="per-alliance field in score_breakdown to compare vs vision "
        "(dotted path). Default: alliance total score.",
    )
    p.add_argument(
        "--show-breakdown",
        metavar="MATCH_KEY",
        default=SHOW_BREAKDOWN or None,
        help="print one match's TBA score_breakdown and exit",
    )
    p.add_argument(
        "--within",
        type=int,
        default=WITHIN,
        help="count an estimate 'correct' if within this many of truth",
    )
    args = p.parse_args()

    truth = load_truth()

    if args.show_breakdown:
        t = truth.get(args.show_breakdown)
        if not t:
            raise SystemExit(f"No truth for {args.show_breakdown}. Run download_matches.py first.")
        print(
            json.dumps(
                {
                    "key": t["key"],
                    "red_score": t["red_score"],
                    "blue_score": t["blue_score"],
                    "score_breakdown": t["score_breakdown"],
                },
                indent=2,
            )
        )
        return

    rows = []
    for est_path in sorted(glob.glob(os.path.join(RESULTS, "*", "estimate.json"))):
        mk = os.path.basename(os.path.dirname(est_path))
        t = truth.get(mk)
        if not t:
            continue
        est = json.load(open(est_path))
        vis = est.get("alliance_scored_from_board", {})
        for alliance in ("blue", "red"):
            v = vis.get(alliance)
            g = tba_value(t, alliance, args.tba_field)
            err = (v - g) if (v is not None and g is not None) else None
            rows.append(
                {
                    "match": mk,
                    "alliance": alliance,
                    "vision": v,
                    "tba": g,
                    "error": err,
                    "abs_error": abs(err) if err is not None else None,
                }
            )

    if not rows:
        raise SystemExit("No results to compare. Run batch_run.py first.")

    with open(os.path.join(HERE, "report.csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["match", "alliance", "vision", "tba", "error", "abs_error"]
        )
        w.writeheader()
        w.writerows(rows)

    paired = [r for r in rows if r["error"] is not None]
    n = len(paired)
    field_label = args.tba_field or "alliance_score"
    summary = {"tba_field": field_label, "pairs": n, "rows_total": len(rows)}
    if n:
        mae = sum(r["abs_error"] for r in paired) / n
        bias = sum(r["error"] for r in paired) / n
        within = sum(1 for r in paired if r["abs_error"] <= args.within) / n
        # Pearson r between vision and tba
        xs = [r["vision"] for r in paired]
        ys = [r["tba"] for r in paired]
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        r = cov / (vx * vy) if vx and vy else None
        summary.update(
            {
                "MAE": round(mae, 2),
                "bias_vision_minus_tba": round(bias, 2),
                f"within_{args.within}_rate": round(within, 3),
                "pearson_r": round(r, 3) if r is not None else None,
            }
        )
    with open(os.path.join(HERE, "report.json"), "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  VISION vs TBA  (field: {field_label})")
    print("=" * 60)
    print(f"  paired alliance-observations: {n} / {len(rows)}")
    if n:
        print(f"  MAE:               {summary['MAE']}")
        print(f"  bias (vis - tba):  {summary['bias_vision_minus_tba']}")
        print(
            f"  within +/-{args.within}:       {summary[f'within_{args.within}_rate'] * 100:.0f}%"
        )
        print(f"  Pearson r:         {summary['pearson_r']}")
    print("  wrote report.csv + report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
