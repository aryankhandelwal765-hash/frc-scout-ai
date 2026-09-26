"""
Visual check of the MULTI-VIEW IDENTITY FUSION.

Renders the three time-aligned views in one frame -- top (overhead) above, revL and
revR below -- and draws, live, the evidence each contributes:

  TOP    each tracked robot's ground point + trail, from trajectory.csv, labelled
         with the team it finally resolved to and HOW (read / rank / none).
  SIDES  whatever scan_side() reads off the reverse cameras at this moment, which
         is the evidence that votes for a track without any detector at all.
  PANEL  the running vote table per track, so you can watch a weak read firm up.

Why it reads trajectory.csv instead of re-tracking: render_match.py calls
model.track() itself, which mints BRAND NEW track ids that do not match the ones in
identity.json, so its identity labels silently never bind. This file uses the same
ids the pipeline used, which is the whole point of a verification render.

PyCharm: set MATCH / START / END below, hit Run.
"""

import csv
import json
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")
TRUTH = os.path.join(HERE, "truth")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
MATCH = "2026hop_qm100"  # a 5-read match; qm27 is the other well-studied one
START = 0.0  # clip start, seconds (0 = start of the trimmed video)
END = 22.0  # clip end, seconds. Auto is ~2-17s, so this covers it.
SIDE_EVERY = 45  # OCR each reverse view every N source frames
OUT = ""  # output path; "" = results/<MATCH>/fusion_<start>-<end>.mp4
# ===========================================================================

BLUE_BGR = (255, 120, 0)
RED_BGR = (0, 80, 255)
GREY = (170, 170, 170)
TRAIL = 60  # trail length, in trajectory samples
PANEL_W = 420  # right-hand evidence panel width, px
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fps_and_stride(match):
    p = os.path.join(RESULTS, match, "tracks.json")
    if os.path.exists(p):
        d = json.load(open(p))
        return float(d.get("fps", 60.0)), int(d.get("stride", 1))
    return 60.0, 1


def _load_traj(match):
    """frame -> [(tid, alliance, cx, cy)], plus tid -> alliance."""
    per_frame = defaultdict(list)
    alliance = {}
    p = os.path.join(RESULTS, match, "trajectory.csv")
    if not os.path.exists(p):
        raise SystemExit(f"No trajectory.csv for {match} -- run the pipeline first.")
    votes = defaultdict(lambda: defaultdict(int))
    with open(p) as f:
        for r in csv.DictReader(f):
            tid, fr = int(r["tid"]), int(r["frame"])
            per_frame[fr].append((tid, r["alliance"], float(r["cx"]), float(r["cy"])))
            votes[tid][r["alliance"]] += 1
    for tid, v in votes.items():
        alliance[tid] = max(v, key=v.get)
    return per_frame, alliance


def _load_identity(match):
    p = os.path.join(RESULTS, match, "identity.json")
    if not os.path.exists(p):
        return {}, {}
    d = json.load(open(p))
    tracks = d.get("tracks", {})
    team = {int(k): v["team"] for k, v in tracks.items() if v.get("team")}
    votes = {int(k): v.get("votes", {}) for k, v in tracks.items()}
    return team, votes


def _load_auto(match):
    """tid -> (team, source, station) for the robots in the auto book."""
    p = os.path.join(RESULTS, match, "auto_paths.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p)).get("teams", {})
    return {o["tid"]: (o.get("team"), o.get("source"), o.get("station")) for o in d.values()}


def _side_hits(match, view, candidates, favour, lo, hi, every):
    """{frame: [(team, conf)]} read straight off a reverse view, no detector."""
    import read_numbers as rn

    path = os.path.join(VIDEOS, view, f"{match}.mp4")
    if not os.path.exists(path):
        return {}
    out = {}
    cap = cv2.VideoCapture(path)
    f = lo - (lo % every)
    while f <= hi:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        # scan_side returns (team, box, score, raw); keep just team + score.
        hits = rn.scan_side(frame, candidates, favour=favour) or []
        norm = []
        for h in hits:
            if isinstance(h, (list, tuple)):
                norm.append((h[0], h[2] if len(h) > 2 else None))
            else:
                norm.append((h, None))
        if norm:
            out[f] = norm
        f += every
    cap.release()
    return out


def _put(img, text, org, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def run(match=MATCH, start=START, end=END, side_every=SIDE_EVERY, out=OUT):
    import read_numbers as rn
    from trajectories import field_crop

    top_path = os.path.join(VIDEOS, f"{match}.mp4")
    if not os.path.exists(top_path):
        raise SystemExit(f"No video: {top_path}")

    fps, stride = _fps_and_stride(match)
    lo, hi = int(start * fps), int(end * fps)
    per_frame, tid_alliance = _load_traj(match)
    ident_team, ident_votes = _load_identity(match)
    auto = _load_auto(match)

    candidates = rn.match_candidates(match) or []
    blue_c = rn.alliance_candidates(match, "blue") or []
    red_c = rn.alliance_candidates(match, "red") or []
    print(
        f"{match}: {len(candidates)} candidate teams, "
        f"{len(ident_team)} tracks with a read, {len(auto)} robots in the auto book"
    )

    print("scanning reverse views for reads (no detector needed) ...")
    side = {
        "revL": _side_hits(match, "revL", candidates, tuple(blue_c), lo, hi, side_every),
        "revR": _side_hits(match, "revR", candidates, tuple(red_c), lo, hi, side_every),
    }
    print(
        f"  revL hits on {len(side['revL'])} sampled frames, "
        f"revR hits on {len(side['revR'])} sampled frames"
    )

    cap = cv2.VideoCapture(top_path)
    capL = cv2.VideoCapture(os.path.join(VIDEOS, "revL", f"{match}.mp4"))
    capR = cv2.VideoCapture(os.path.join(VIDEOS, "revR", f"{match}.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1, y1, x2, y2 = field_crop((H, W))

    okL, fL = capL.read()
    sh = fL.shape[:2] if okL else (254, 874)
    side_h = int(sh[0] * (W / 2.0) / sh[1])  # each side view fills half the width
    out_w, out_h = W + PANEL_W, H + side_h

    out = out or os.path.join(RESULTS, match, f"fusion_{int(start)}-{int(end)}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, stride), (out_w, out_h))

    trails = defaultdict(lambda: deque(maxlen=TRAIL))
    last_side = {"revL": (None, []), "revR": (None, [])}
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    capL.set(cv2.CAP_PROP_POS_FRAMES, lo)
    capR.set(cv2.CAP_PROP_POS_FRAMES, lo)

    auto_lo, auto_hi = int(2.0 * fps), int(17.0 * fps)
    f = lo
    written = 0
    while f <= hi:
        ok, frame = cap.read()
        okL, frL = capL.read()
        okR, frR = capR.read()
        if not ok:
            break
        if stride > 1 and (f - lo) % stride != 0:
            f += 1
            continue

        canvas = np.zeros((out_h, out_w, 3), np.uint8)
        canvas[0:H, 0:W] = frame

        # ---- TOP: tracks with their fused identity -------------------------
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), 1)
        for tid, al, cx, cy in per_frame.get(f, []):
            trails[tid].append((int(cx), int(cy)))
            col = BLUE_BGR if al == "blue" else RED_BGR
            pts = list(trails[tid])
            if len(pts) >= 2:
                cv2.polylines(
                    canvas[y1:y2, x1:x2], [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA
                )
            px, py = x1 + int(cx), y1 + int(cy)
            cv2.circle(canvas, (px, py), 6, col, -1)
            cv2.circle(canvas, (px, py), 6, (255, 255, 255), 1)

            team, source, station = auto.get(tid, (None, None, None))
            if team:
                label = f"{team.replace('frc', '#')} [{source}]"
                lcol = (90, 255, 90) if source == "read" else (0, 200, 255)
            elif tid in ident_team:
                label = f"#{ident_team[tid]} [read]"
                lcol = (90, 255, 90)
            else:
                label = f"t{tid}"
                lcol = GREY
            _put(canvas, label, (px + 9, py - 9), lcol, 0.52, 1)

        # ---- SIDES: raw reverse views + what the OCR pulled off them --------
        for i, (name, frv, okv) in enumerate((("revL", frL, okL), ("revR", frR, okR))):
            xoff = i * (W // 2)
            if okv:
                v = cv2.resize(frv, (W // 2, side_h))
                canvas[H : H + side_h, xoff : xoff + (W // 2)] = v
            hits = side[name].get(f)
            if hits:
                last_side[name] = (f, hits)
            lf, lh = last_side[name]
            fresh = lf is not None and (f - lf) < side_every
            _put(
                canvas,
                f"{name}  ({'BLUE' if name == 'revL' else 'RED'} side)",
                (xoff + 10, H + 22),
                (255, 255, 255),
                0.6,
                2,
            )
            if lh:
                txt = "  ".join(f"#{t}" for t, _ in lh)
                _put(
                    canvas,
                    f"read: {txt}",
                    (xoff + 10, H + 48),
                    (90, 255, 90) if fresh else (120, 140, 120),
                    0.62,
                    2,
                )
            else:
                _put(canvas, "read: -", (xoff + 10, H + 48), GREY, 0.6, 2)

        # ---- PANEL: the fusion evidence table -------------------------------
        px0 = W
        cv2.rectangle(canvas, (px0, 0), (out_w, out_h), (24, 24, 24), -1)
        y = 26
        _put(canvas, match, (px0 + 12, y), (255, 255, 255), 0.62, 2)
        y += 24
        in_auto = auto_lo <= f <= auto_hi
        _put(
            canvas,
            f"frame {f}   t={f / fps:5.1f}s" + ("   [AUTO]" if in_auto else ""),
            (px0 + 12, y),
            (90, 255, 90) if in_auto else GREY,
            0.5,
        )
        y += 26
        _put(canvas, "AUTO BOOK  (team / how)", (px0 + 12, y), (255, 255, 255), 0.52, 1)
        y += 20
        for tid, (team, source, station) in sorted(auto.items(), key=lambda kv: kv[1][2] or ""):
            col = (90, 255, 90) if source == "read" else (0, 200, 255)
            live = f in {fr for fr in (f,)} and any(t == tid for t, *_ in per_frame.get(f, []))
            mark = ">" if live else " "
            _put(
                canvas,
                f"{mark} {station or '?':6s} {str(team).replace('frc', '#'):8s} {source or '-'}",
                (px0 + 12, y),
                col,
                0.5,
            )
            y += 19
        y += 10
        _put(canvas, "PER-TRACK VOTES (identity.json)", (px0 + 12, y), (255, 255, 255), 0.52, 1)
        y += 20
        live_tids = {t for t, *_ in per_frame.get(f, [])}
        shown = 0
        for tid in sorted(ident_votes, key=lambda t: (t not in live_tids, t)):
            v = ident_votes[tid]
            if not v:
                continue
            top = sorted(v.items(), key=lambda kv: -kv[1])[:3]
            txt = " ".join(f"{t}:{n:g}" for t, n in top)
            col = (255, 255, 255) if tid in live_tids else (120, 120, 120)
            _put(canvas, f"t{tid:<4d} {txt}", (px0 + 12, y), col, 0.46)
            y += 17
            shown += 1
            if y > out_h - 30 or shown > 18:
                break

        vw.write(canvas)
        written += 1
        f += 1

    cap.release()
    capL.release()
    capR.release()
    vw.release()
    print(f"wrote {written} frames -> {out}")
    return out


if __name__ == "__main__":
    run()
