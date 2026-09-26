"""
Render a WHOLE match: broadcast top view with tracked robots, a live top-down field,
and a HUD that names the MATCH phase from the game manual's real timing.

Phases come from the 2026 REBUILT manual, not guessed:
  AUTO        20s
  TRANSITION  10s   (TELEOP timer 2:20 -> 2:10)
  SHIFT 1-4   25s each
  END GAME    30s   (0:30 -> 0:00)
which is exactly what TBA reports as hubScore.shift1Count..shift4Count,
transitionCount and endgameCount -- so the bar along the bottom lines up with the
official per-shift scoring.

Identity is locked over WHOLE-MATCH chains, so a robot keeps its label when
ByteTrack drops and re-acquires it. Labels state how the team was determined
(read / elim / rank), because a rank label is an unverified guess and should never
look as solid as a bumper read.

PyCharm: set MATCH, click Run.
"""

import argparse
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
import auto_paths as ap
import stitch
import topdown as td

VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
MATCH = "2026hop_qm20"
STRIDE = 2  # 2 = 30fps output from 60fps source. 1 = every frame (huge).
WIDTH = 1600  # output width; height follows
TRAIL = 90  # motion trail length, in samples
OUT = ""  # "" = results/<MATCH>/full_match_render.mp4
# ===========================================================================

# (name, seconds from AUTO start, colour) -- straight from the manual timing
PHASES = [
    ("AUTO", 0.0, 20.0, (90, 255, 90)),
    ("TRANSITION", 20.0, 30.0, (200, 200, 200)),
    ("SHIFT 1", 30.0, 55.0, (255, 190, 90)),
    ("SHIFT 2", 55.0, 80.0, (255, 160, 60)),
    ("SHIFT 3", 80.0, 105.0, (255, 130, 40)),
    ("SHIFT 4", 105.0, 130.0, (255, 100, 20)),
    ("END GAME", 130.0, 160.0, (80, 80, 255)),
]

BLUE_BGR = (255, 120, 0)
RED_BGR = (0, 80, 255)
SRC_COL = {
    "read": (90, 255, 90),
    "elim": (255, 220, 90),
    "rank": (0, 165, 255),
    None: (170, 170, 170),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put(img, text, org, col=(255, 255, 255), scale=0.6, th=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, col, th, cv2.LINE_AA)


def phase_at(sec):
    for name, a, b, col in PHASES:
        if a <= sec < b:
            return name, col
    return ("PRE" if sec < 0 else "POST"), (150, 150, 150)


def build(match=MATCH, stride=STRIDE, width=WIDTH, out=""):
    vid = os.path.join(VIDEOS, f"{match}.mp4")
    if not os.path.exists(vid):
        raise SystemExit(f"No video: {vid}")
    apf = os.path.join(RESULTS, match, "auto_paths.json")
    tracks = ap._load_trajectory(match)
    fps = 60.0
    tj = os.path.join(RESULTS, match, "tracks.json")
    if os.path.exists(tj):
        fps = json.load(open(tj)).get("fps", 60.0)

    start = ap._detected_start(match)
    if start is None:
        start = int(ap.AUTO_START_SEC * fps)

    # whole-match identity: chain fragments, then a read anywhere locks the chain
    al_of = {t: ap._voted_alliance(match, t, r) for t, r in tracks.items()}
    last = max(r[0] for rows in tracks.values() for r in rows)
    chains = stitch.chains(match, tracks, al_of, fps=fps, window=(0, last))
    locked, _ = stitch.lock_identity(chains, ap._read_identity(match), tracks)
    tid_team = dict(locked)

    # the auto book knows HOW each station's team was decided -- reuse that label
    src_by_team = {}
    if os.path.exists(apf):
        for _st, o in json.load(open(apf)).get("teams", {}).items():
            if o.get("team"):
                src_by_team[o["team"].replace("frc", "")] = o.get("source")

    # steal excursions for this match, so the render shows the one thing vision
    # produces that neither TBA nor a hand scout does: timed round trips into the
    # opponent half. Each is (out_frame, home_frame, team, ok).
    steal_ev = []
    sp = os.path.join(HERE, "steals.csv")
    if os.path.exists(sp):
        import csv as _csv

        with open(sp) as _f:
            for _r in _csv.DictReader(_f):
                if _r["match"] != match:
                    continue
                steal_ev.append(
                    (
                        int(_r["out_frame"]),
                        int(_r["home_frame"]),
                        _r.get("team", ""),
                        int(_r["successful"]),
                        float(_r["max_depth_in"]),
                    )
                )
    steal_ev.sort()

    per_frame = defaultdict(list)
    for tid, rows in tracks.items():
        for f, _t, al, cx, cy, _c in rows:
            per_frame[f].append((tid, al, cx, cy))

    cap = cv2.VideoCapture(vid)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    from trajectories import field_crop

    x1, y1, x2, y2 = field_crop((H, W, 3))

    scale = width / float(W)
    vw_h = int(H * scale)
    mini = td.draw_field(blue_left=True)
    mini_w = width
    mini_h = int(mini.shape[0] * mini_w / mini.shape[1])
    hud_h = 96
    out_h = vw_h + mini_h + hud_h

    out = out or os.path.join(RESULTS, match, "full_match_render.mp4")
    writer = cv2.VideoWriter(
        out, cv2.VideoWriter_fourcc(*"mp4v"), fps / max(1, stride), (width, out_h)
    )
    mapper = td.load_mapper()
    trails = defaultdict(lambda: deque(maxlen=TRAIL))

    f = 0
    written = 0
    while f < total:
        ok, frame = cap.read()
        if not ok:
            break
        if f % stride:
            f += 1
            continue
        sec = (f - start) / fps
        pname, pcol = phase_at(sec)

        canvas = np.zeros((out_h, width, 3), np.uint8)
        top = cv2.resize(frame, (width, vw_h))
        canvas[0:vw_h] = top

        live = per_frame.get(f, [])
        board = mini.copy()
        for tid, al, cx, cy in live:
            col = BLUE_BGR if al == "blue" else RED_BGR
            px = int((x1 + cx) * scale)
            py = int((y1 + cy) * scale)
            trails[tid].append((px, py))
            pts = list(trails[tid])
            if len(pts) >= 2:
                cv2.polylines(canvas[0:vw_h], [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 7, col, -1)
            cv2.circle(canvas, (px, py), 7, (255, 255, 255), 1)
            team = tid_team.get(tid)
            src = src_by_team.get(team) if team else None
            label = f"#{team}" if team else f"t{tid}"
            _put(
                canvas,
                label + (f" [{src}]" if src else ""),
                (px + 10, py - 10),
                SRC_COL.get(src, (170, 170, 170)),
                0.5,
                1,
            )
            if mapper is not None:
                X, Y = mapper([(cx, cy)])[0]
                if 0 <= X < td.CW and 0 <= Y < td.CH:
                    cv2.circle(board, (int(X), int(Y)), 13, col, -1)
                    cv2.circle(board, (int(X), int(Y)), 13, (255, 255, 255), 2)
                    if team:
                        _put(
                            board,
                            f"#{team}",
                            (int(X) + 16, int(Y) - 12),
                            SRC_COL.get(src, (60, 60, 60)),
                            0.6,
                            2,
                        )
        canvas[vw_h : vw_h + mini_h] = cv2.resize(board, (mini_w, mini_h))

        # ---- HUD: clock, phase, and a phase bar that matches TBA's shift fields
        y0 = vw_h + mini_h
        cv2.rectangle(canvas, (0, y0), (width, out_h), (18, 18, 18), -1)
        _put(canvas, match, (14, y0 + 30), (255, 255, 255), 0.72, 2)
        _put(canvas, f"t={sec:6.1f}s", (250, y0 + 30), (200, 200, 200), 0.66, 2)
        _put(canvas, pname, (410, y0 + 30), pcol, 0.78, 2)
        _put(
            canvas,
            f"{len(live)} robots tracked",
            (620, y0 + 30),
            (200, 200, 200) if len(live) == 6 else (0, 165, 255),
            0.6,
            2,
        )
        _put(
            canvas,
            "green=read  yellow=elimination  orange=UNVERIFIED rank guess",
            (900, y0 + 30),
            (150, 150, 150),
            0.5,
            1,
        )
        # active steals: a robot currently in the opponent half on a round trip
        live_st = [e for e in steal_ev if e[0] <= f <= e[1]]
        if live_st:
            txt = "  ".join(
                f"#{t or '?'} STEAL {d:.0f}in" + (" OK" if ok else "")
                for _a, _b, t, ok, d in live_st[:3]
            )
            _put(canvas, txt, (14, vw_h + mini_h - 14), (60, 230, 255), 0.62, 2)
        done = sum(1 for e in steal_ev if e[1] <= f)
        _put(
            canvas,
            f"steals so far: {done}/{len(steal_ev)}",
            (width - 250, y0 + 30),
            (60, 230, 255),
            0.55,
            2,
        )
        bx0, bx1 = 14, width - 14
        by = y0 + 58
        span = PHASES[-1][2]
        for name, a, b, col in PHASES:
            xa = int(bx0 + (bx1 - bx0) * a / span)
            xb = int(bx0 + (bx1 - bx0) * b / span)
            cv2.rectangle(canvas, (xa, by), (xb - 2, by + 22), col, -1)
            _put(canvas, name.replace("SHIFT ", "S"), (xa + 5, by + 17), (20, 20, 20), 0.42, 1)
        if 0 <= sec <= span:
            cur = int(bx0 + (bx1 - bx0) * sec / span)
            cv2.line(canvas, (cur, by - 5), (cur, by + 27), (255, 255, 255), 3)

        writer.write(canvas)
        written += 1
        f += 1

    cap.release()
    writer.release()
    print(f"{written} frames -> {out}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", default=MATCH)
    p.add_argument("--stride", type=int, default=STRIDE)
    a = p.parse_args()
    build(a.match, a.stride)


if __name__ == "__main__":
    main()
