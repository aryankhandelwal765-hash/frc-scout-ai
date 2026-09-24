"""
Auto-period path extraction + robot->team classification, per match.

Pipeline per match (needs trajectory.csv from trajectories.py + calib.json from
field_calibrate.py + truth/<match>.json from download_matches.py):

  1. Find the auto window  [start, start + AUTO_SECONDS].
  2. For each tracked robot, take its start position inside that window.
  3. CLASSIFY: greedily match each alliance's robot start positions to that
     alliance's 3 station anchors -> station -> TBA team number.
        (TBA gives team_keys ordered by station 1,2,3.)
  4. Emit each team's auto path, and draw them on the reference frame.

Output -> results/<match>/:
  auto_paths.json : team_key -> {station, tid, alliance, path:[[t,cx,cy],...]}
  auto_paths.png  : reference frame with each team's auto path + team number

  python auto_paths.py --match 2026hop_qm1
  python auto_paths.py --match 2026hop_qm1 --start-frame 190 --auto-seconds 15
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np

import stitch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TRUTH = os.path.join(HERE, "truth")
VIDEOS = os.path.join(HERE, "videos")
CALIB = os.path.join(HERE, "calib.json")

BLUE_BGR = (255, 120, 0)
RED_BGR = (0, 80, 255)
# Auto is 20 SECONDS, read straight off the broadcast timer -- it counts 20 down to
# 1, then jumps to 2:20 for teleop. This was 15.0, which silently cut the last
# quarter off every auto path.
AUTO_SECONDS = 20.0
# Fallback only. preprocess trims on a timer sampled every 2s, so auto's position in
# the cleaned video is PAD +- 2s, NOT a constant -- measured across 128 matches it
# lands anywhere from frame -20 to +100 (median 60), never the 120 this assumed.
# auto_start.py reads the countdown per match and writes auto_start.json; use that
# when it exists and only fall back to this constant when it does not.
AUTO_START_SEC = 1.0
AUTO_START_JSON = os.path.join(HERE, "auto_start.json")
# Which end of the alliance wall is station 1? "ltr" = leftmost robot is station 1,
# "rtl" = rightmost. Flip if the team labels come out swapped vs the real roster.
ALLIANCE_ORDER = {"blue": "ltr", "red": "rtl"}


def _load_calib():
    if not os.path.exists(CALIB):
        raise SystemExit("No calib.json -- run field_calibrate.py first.")
    return json.load(open(CALIB))


def _load_trajectory(match):
    path = os.path.join(RESULTS, match, "trajectory.csv")
    if not os.path.exists(path):
        raise SystemExit(f"No trajectory.csv for {match} -- run trajectories.py first.")
    tracks = defaultdict(list)   # tid -> [(frame, t, alliance, cx, cy, conf)]
    with open(path) as f:
        for r in csv.DictReader(f):
            tracks[int(r["tid"])].append((int(r["frame"]), float(r["t"]), r["alliance"],
                                          float(r["cx"]), float(r["cy"]), float(r["conf"])))
    for tid in tracks:
        tracks[tid].sort()
    return tracks


def _voted_alliance(match, tid, rows):
    b = sum(1 for x in rows if x[2] == "blue")
    return "blue" if b >= len(rows) - b else "red"


def _match_start_frame(match, fps):
    """Best-effort auto start: first frame the scoreboard ball count starts to
    climb (from board_series.csv if present). Returns frame index or None."""
    bs = os.path.join(RESULTS, match, "board_series.csv")
    if not os.path.exists(bs):
        return None
    prev = {"blue": None, "red": None}
    with open(bs) as f:
        for r in csv.DictReader(f):
            for a in ("blue", "red"):
                v = r.get(f"{a}_balls")
                v = int(v) if v not in (None, "", "None") else None
                if v is not None:
                    if prev[a] is not None and v > prev[a]:
                        return int(r["frame"])
                    prev[a] = v
    return None


def _greedy_assign(track_starts, anchors):
    """track_starts: {tid:(x,y)}; anchors: {station:(x,y)} (same alliance, <=3 each).
    Greedily pair nearest (tid, station). Returns {tid: station}."""
    pairs = []
    for tid, (tx, ty) in track_starts.items():
        for st, (ax, ay) in anchors.items():
            pairs.append((((tx - ax) ** 2 + (ty - ay) ** 2) ** 0.5, tid, st))
    pairs.sort()
    used_t, used_s, out = set(), set(), {}
    for d, tid, st in pairs:
        if tid in used_t or st in used_s:
            continue
        out[tid] = st
        used_t.add(tid); used_s.add(st)
    return out


def _detected_start(match):
    """This match's auto-start frame from auto_start.py, or None if it never ran."""
    if not os.path.exists(AUTO_START_JSON):
        return None
    try:
        d = json.load(open(AUTO_START_JSON)).get(match)
    except Exception:
        return None
    return int(d["start_frame"]) if d else None


def _read_identity(match):
    """tid -> team, from identity.json (bumper numbers actually READ). This is
    direct evidence and outranks the rank-order guess."""
    p = os.path.join(RESULTS, match, "identity.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p)).get("tracks", {})
    return {int(k): v["team"] for k, v in d.items() if v.get("team")}




def run(match, start_frame=None, auto_seconds=AUTO_SECONDS, min_track_frames=5):
    calib = json.load(open(CALIB)) if os.path.exists(CALIB) else {}
    truth = json.load(open(os.path.join(TRUTH, f"{match}.json")))
    teams = {"blue": truth.get("blue_teams") or [], "red": truth.get("red_teams") or []}

    read_id = _read_identity(match)
    tracks = _load_trajectory(match)
    fps = 60.0
    tj = os.path.join(RESULTS, match, "tracks.json")
    if os.path.exists(tj):
        fps = json.load(open(tj)).get("fps", 60.0)

    # Where auto actually begins, READ off this match's own timer countdown.
    # A per-match value matters: the trim is only accurate to +-2s, so a shared
    # constant puts the window up to two seconds off and every start position with it.
    src = "detected"
    if start_frame is None:
        start_frame = _detected_start(match)
        if start_frame is None:
            start_frame, src = int(AUTO_START_SEC * fps), "assumed"
    else:
        src = "given"
    start_frame = max(0, start_frame)
    end_frame = start_frame + int(auto_seconds * fps)

    # STITCH first. ByteTrack splits one robot across many ids, so a bumper read on
    # an early fragment used to die with that fragment and the robot got a rank-order
    # name for the rest of auto (2026hop_qm100: #1706 read on track 19, then called
    # 3035 from track 28 onward -- same robot). Chain the fragments, then a read
    # anywhere on a chain locks the whole chain.
    al_of = {tid: _voted_alliance(match, tid, rows) for tid, rows in tracks.items()}
    chain_list = stitch.chains(match, tracks, al_of, fps=fps,
                               window=(start_frame, end_frame))
    locked, _per_chain = stitch.lock_identity(chain_list, read_id)

    # per chain: every fragment's rows inside the auto window, and a start position
    per_alliance_starts = {"blue": {}, "red": {}}
    auto_rows = {}
    chain_of = {}
    for ci, ch in enumerate(chain_list):
        win = sorted(r for tid in ch for r in tracks.get(tid, [])
                     if start_frame <= r[0] <= end_frame)
        if len(win) < min_track_frames:
            continue
        al = al_of.get(ch[0])
        if al not in per_alliance_starts:
            continue
        sx = float(np.median([r[3] for r in win[:5]]))
        sy = float(np.median([r[4] for r in win[:5]]))
        per_alliance_starts[al][ci] = (sx, sy)
        auto_rows[ci] = (al, win)
        chain_of[ci] = ch

    out = {}
    dbg = {"match": match, "start_frame": start_frame, "end_frame": end_frame,
           "fps": fps, "start_source": src, "auto_seconds": auto_seconds}
    for al in ("blue", "red"):
        # classify by SPATIAL ORDER along the alliance wall (no fixed anchors):
        # sort the detected robots by x, assign station 1,2,3 in that order -> team.
        # ALLIANCE_ORDER flips which end is station 1 if labels come out swapped.
        starts = per_alliance_starts[al]
        # take the 3 most-PERSISTENT tracks (most frames in the auto window) so a
        # flickery false track can't displace a real robot, then rank those by x
        cand = sorted(starts, key=lambda t: -len(auto_rows[t][1]))[:3]
        order = sorted(cand, key=lambda t: starts[t][0],
                       reverse=(ALLIANCE_ORDER.get(al) == "rtl"))
        # rank-based station only holds when all three robots are tracked; with a
        # missing (or extra) one the ranks shift and every team number is wrong.
        confident = len(order) == 3

        # TWO PASSES, reads first. A read and the rank guess can name the SAME team
        # for two different tracks (they disagree about which robot it is). Reads are
        # direct evidence, so they claim their team outright and rank order is then
        # only allowed to hand out teams nobody read.
        assigned = {}                       # chain -> (team, source)
        # A read is only admissible if TBA says that team is on THIS alliance.
        # Bumper colour occasionally calls a track for the wrong alliance, and the
        # read then rides along -- measured 3 of 360 reads (0.8%) across 112
        # matches, each one also forcing a duplicate team elsewhere in the match.
        roster = {t.replace("frc", "") for t in teams[al]}
        for ci in order[:3]:
            t = next((locked[tid] for tid in chain_of[ci] if tid in locked), None)
            if t and t in roster:
                assigned[ci] = (f"frc{t}", "read")

        claimed = {t for t, _ in assigned.values()}
        spare = [t for t in teams[al] if t not in claimed]
        rest = [ci for ci in order[:3] if ci not in assigned]

        # ELIMINATION. If every robot but one is already named by a read, the last
        # one is not a guess -- it is the only team left on the roster. That is
        # certain and owes nothing to ALLIANCE_ORDER, which is still unvalidated.
        # Only when TWO or more are unnamed does left-to-right order actually decide
        # anything, and only then is the answer a genuine rank-order guess.
        if len(rest) == 1 and len(spare) == 1:
            assigned[rest[0]] = (spare[0], "elim")
        else:
            for ci in rest:
                if confident and spare:
                    assigned[ci] = (spare.pop(0), "rank")
                else:
                    assigned[ci] = (None, "none")

        for idx, ci in enumerate(order[:3]):
            st = f"{al}{idx + 1}"
            team, source = assigned[ci]
            al2, win = auto_rows[ci]
            # key by STATION, never by team: keying by team silently dropped a robot
            # whenever a read and the rank guess landed on the same number.
            # `tids` is the whole stitched chain; `tid` stays the primary fragment so
            # anything still joining on a single id keeps working.
            out[st] = {"station": st, "tid": chain_of[ci][0], "tids": chain_of[ci],
                       "alliance": al, "team": team, "source": source,
                       "path": [[round(t, 3), cx, cy] for fr, t, a, cx, cy, cf in win]}

    outdir = os.path.join(RESULTS, match)
    with open(os.path.join(outdir, "auto_paths.json"), "w") as f:
        json.dump({"debug": dbg, "teams": out}, f, indent=2)
    _plot(match, calib, out, outdir)
    print(f"  {match}: auto [{start_frame}-{end_frame}] ({src})  classified "
          f"{len(out)} robots -> {[o['team'] for o in out.values()]}")
    return out


def _team_label(team_key):
    return team_key.replace("frc", "#") if team_key and team_key.startswith("frc") else str(team_key)


def _plot(match, calib, teams_out, outdir):
    ref = calib.get("reference_frame")
    img = cv2.imread(ref) if ref and os.path.exists(ref) else None
    if img is None:
        h, w = calib["crop"][3] - calib["crop"][1], calib["crop"][2] - calib["crop"][0]
        img = np.full((h, w, 3), 40, np.uint8)
    # draw station anchors faintly (if any were calibrated -- optional now)
    for st, (x, y) in (calib.get("stations") or {}).items():
        col = BLUE_BGR if st.startswith("blue") else RED_BGR
        cv2.drawMarker(img, (int(x), int(y)), col, cv2.MARKER_CROSS, 12, 1)
    for key, o in teams_out.items():
        col = BLUE_BGR if o["alliance"] == "blue" else RED_BGR
        pts = np.array([[int(cx), int(cy)] for t, cx, cy in o["path"]], np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[0]), 5, (255, 255, 255), -1)
            cv2.circle(img, tuple(pts[0]), 5, col, 2)
            cv2.putText(img, _team_label(o["team"]), (pts[0][0] + 6, pts[0][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.putText(img, f"{match} AUTO paths", (8, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(outdir, "auto_paths.png"), img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", required=True, help="match key, e.g. 2026hop_qm1")
    p.add_argument("--start-frame", type=int, help="override auto start frame")
    p.add_argument("--auto-seconds", type=float, default=AUTO_SECONDS)
    args = p.parse_args()
    run(args.match, start_frame=args.start_frame, auto_seconds=args.auto_seconds)


if __name__ == "__main__":
    main()
