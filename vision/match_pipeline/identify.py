"""
Per-track IDENTITY by voting on read bumper numbers (not rank-order guessing).

Two independent evidence sources, fused:
  TOP view   -- for each tracked robot, read the bumper inside its own box, so a
                read attaches directly to a track id.
  SIDE views -- OCR the whole reverse frame (no detector needed) and snap to the
                six teams TBA says are playing. A side hit is stamped with a time,
                so it votes for whichever track is alive at that moment, on the
                matching alliance.

Every read is scored against ONLY the six known teams, so partial reads still
resolve ("5345" -> 5348). Votes accumulate across the whole match, which is what
turns a ~16%-per-frame read rate into a confident per-track identity.

Output -> results/<match>/identity.json
  { tid: {alliance, team, confidence, votes:{team:n}, source} }

  python identify.py --match 2026hop_qm27
  python identify.py                       # every match with a trajectory
"""

import argparse
import csv
import glob
import json
import os
import sys
import warnings
from collections import defaultdict, Counter

import cv2
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PARENT)
VIDEOS = os.path.join(HERE, "videos")
RESULTS = os.path.join(HERE, "results")

# Sampling budget. A track only needs a handful of confident reads to be called,
# so dense sampling is wasted work: the top view costs one EasyOCR call PER BOX,
# while a side scan costs one call for the WHOLE frame and can return several teams
# at once. So sample the side views hard and the top view sparsely.
TOP_EVERY = 120  # sample the top view every N frames (one OCR call per box)
SIDE_EVERY = 45  # sample each side view every N frames (one call per frame)
SIDE_BONUS = 0.10  # tie-break bonus for the alliance a side camera leans to
MIN_VOTES = 2  # need at least this many votes to call a track identified

# ByteTrack fragments badly -- measured 228 track ids for 6 robots on qm27. A flat
# TOP_EVERY across a ~10k-frame match therefore spends its reads on whichever
# fragments live longest, which are late-match ones, and the short auto-window
# tracks get almost no read attempts at all. Since auto_paths.py is the only
# consumer of identity.json, that meant every read landed on a track auto never
# looks at (identity tids 5202-10018 vs auto tids 28-990: zero overlap).
# So sample the auto window hard. It is only ~900 frames, robots are maximally
# separated there, and bumpers face the cameras.
AUTO_EVERY = 10  # sample stride INSIDE the auto window
# Teleop sampling is budgeted per TRACK, not by stride -- see _sample_frames.
# Tuned on qm27: 6/8/700 gave 45 identified tracks in 247s; 4/20/420 gives 40 in
# 182s with IDENTICAL downstream steal attribution (54%). The extra 5 tracks are
# short fragments that change nothing, so the cheaper setting is the default.
PER_TRACK_SAMPLES = 4  # frames OCR'd per teleop track, spread over its lifetime
MIN_TRACK_FRAMES = 20  # ignore flicker fragments shorter than this
MAX_SAMPLES = 420  # global cap on top-view frames OCR'd per match


def _load_tracks(match):
    """tid -> {alliance, frames:set, pos:{frame:(cx,cy)}} from trajectory.csv."""
    p = os.path.join(RESULTS, match, "trajectory.csv")
    if not os.path.exists(p):
        return {}
    votes = defaultdict(Counter)
    frames = defaultdict(set)
    pos = defaultdict(dict)
    with open(p) as f:
        for r in csv.DictReader(f):
            tid = int(r["tid"])
            fr = int(r["frame"])
            votes[tid][r["alliance"]] += 1
            frames[tid].add(fr)
            pos[tid][fr] = (float(r["cx"]), float(r["cy"]))
    return {
        tid: {"alliance": votes[tid].most_common(1)[0][0], "frames": frames[tid], "pos": pos[tid]}
        for tid in frames
    }


def _auto_window(match, tracks):
    """(lo, hi) frame bounds of the auto period, taken from auto_paths' own
    constants so the two files cannot drift apart."""
    import auto_paths as ap

    fps = 60.0
    tj_path = os.path.join(RESULTS, match, "tracks.json")
    if os.path.exists(tj_path):
        try:
            fps = json.load(open(tj_path)).get("fps", 60.0)
        except Exception:
            pass
    lo = int(ap.AUTO_START_SEC * fps)
    return lo, lo + int(ap.AUTO_SECONDS * fps)


def _sample_frames(tracks, auto_lo, auto_hi):
    """Frames to OCR, budgeted PER TRACK rather than by a uniform stride.

    A uniform stride spends its samples where tracks happen to be long, which in
    practice means the auto window and whatever survives in teleop. Measured on
    qm27: of 137 teleop-only tracks, 64% got exactly ONE sample -- and one sample
    can never identify anything, because MIN_VOTES is 2. So teleop identity was
    structurally impossible for most tracks, not merely unlikely.

    Instead give every track its own small budget of frames spread across its
    lifetime, then take the union. Because 6 robots share most frames, the union
    is far smaller than budget x tracks. Longest tracks are served first so that
    if the global cap bites, it drops flickery fragments rather than real robots.

    Still snapped to frames that appear in trajectory.csv -- that file is written
    with a stride, so an arithmetic step can land between logged frames and match
    nothing.
    """
    keep = set()
    # auto stays densely sampled: it is where the robots are separated and facing
    # the cameras, and it is what the auto book consumes.
    logged = sorted({f for t in tracks.values() for f in t["pos"]})
    last = None
    for f in logged:
        if auto_lo <= f <= auto_hi and (last is None or f - last >= AUTO_EVERY):
            keep.add(f)
            last = f

    order = sorted(tracks.items(), key=lambda kv: -len(kv[1]["pos"]))
    for _tid, t in order:
        fr = sorted(f for f in t["pos"] if f > auto_hi)
        if len(fr) < MIN_TRACK_FRAMES:
            continue
        idx = np.linspace(0, len(fr) - 1, PER_TRACK_SAMPLES).astype(int)
        keep.update(fr[i] for i in set(idx.tolist()))
        if len(keep) >= MAX_SAMPLES:
            break
    return sorted(keep)


def _top_reads(match, tracks, candidates, model, rn, frames=None):
    """Read the bumper inside each tracked box -> votes keyed by track id."""
    import trajectories as tj

    vid = os.path.join(VIDEOS, f"{match}.mp4")
    if not os.path.exists(vid):
        return defaultdict(Counter)
    out = defaultdict(Counter)
    cap = cv2.VideoCapture(vid)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames is None:
        frames = list(range(0, total, TOP_EVERY))
    for f in frames:
        if f >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        cx0, cy0, cx1, cy1 = tj.field_crop(frame.shape)
        crop = frame[cy0:cy1, cx0:cx1]
        res = model.predict(crop, conf=0.3, imgsz=960, agnostic_nms=True, verbose=False)[0]
        # Re-running the tracker would mint NEW ids that don't match trajectory.csv,
        # so instead bind each detection to the existing track whose logged ground
        # point is nearest at this frame.
        near = {tid: t["pos"][f] for tid, t in tracks.items() if f in t["pos"]}
        if res.boxes is not None:
            for b in res.boxes:
                bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0]]
                gx, gy = (bx1 + bx2) / 2.0, by2
                tid, best = None, 1e9
                for k, (px, py) in near.items():
                    d = (gx - px) ** 2 + (gy - py) ** 2
                    if d < best:
                        best, tid = d, k
                if tid is None or best > 120**2:
                    continue
                if sum(out[tid].values()) >= 4:
                    continue  # already well-evidenced; skip the OCR
                team, conf = rn.read_bumper(crop, (bx1, by1, bx2, by2), candidates)
                if team:
                    out[tid][team] += 1
    cap.release()
    return out


def _side_reads(match, candidates, favour_by_view, rn, every=SIDE_EVERY):
    """OCR both reverse views -> [(frame, team)] time-stamped votes."""
    hits = []
    for view, favour in favour_by_view.items():
        vid = os.path.join(VIDEOS, view, f"{match}.mp4")
        if not os.path.exists(vid):
            continue
        cap = cv2.VideoCapture(vid)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f = 0
        while f < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                break
            for team, bb, sc, raw in rn.scan_side(
                frame, candidates, favour=favour, bonus=SIDE_BONUS
            ):
                hits.append((f, team))
            f += every
        cap.release()
    return hits


def run(match, model=None):
    import read_numbers as rn
    from ultralytics import YOLO
    import trajectories as tj

    tracks = _load_tracks(match)
    if not tracks:
        return None
    candidates = rn.match_candidates(match)
    if not candidates:
        return None
    team_alliance = {}
    for al in ("blue", "red"):
        for t in rn.alliance_candidates(match, al):
            team_alliance[t] = al

    model = model or YOLO(tj._newest_weights())
    votes = defaultdict(Counter)

    # 1. top view -- reads attach straight to a track id. Dense inside auto,
    #    sparse elsewhere (see AUTO_EVERY).
    auto_lo, auto_hi = _auto_window(match, tracks)
    frames = _sample_frames(tracks, auto_lo, auto_hi)
    for tid, c in _top_reads(match, tracks, candidates, model, rn, frames=frames).items():
        votes[tid].update(c)

    # 2. side views -- a read is stamped with a frame, so it votes for the tracks
    #    alive at that moment on the SAME alliance as the team read.
    for f, team in _side_reads(
        match,
        candidates,
        {
            "revL": rn.alliance_candidates(match, "blue"),
            "revR": rn.alliance_candidates(match, "red"),
        },
        rn,
    ):
        al = team_alliance.get(team)
        live = [
            tid
            for tid, t in tracks.items()
            if t["alliance"] == al
            and any(abs(fr - f) <= SIDE_EVERY for fr in (min(t["frames"]), max(t["frames"])))
            or (t["alliance"] == al and f in t["frames"])
        ]
        if len(live) == 1:  # unambiguous -> full vote
            votes[live[0]][team] += 1
        elif live:  # ambiguous -> spread a partial vote
            for tid in live:
                votes[tid][team] += 1.0 / len(live)

    # Resolve in TWO INDEPENDENT TIERS, auto-window tracks first.
    #
    # "one team can only be one track" is right within a single moment but wrong
    # across a whole match: 6 robots produce ~228 track ids, so the same team
    # legitimately owns many fragments. Enforcing it globally let long late-match
    # fragments claim all six teams and starved the short auto-window tracks --
    # the exact tracks auto_paths.py consumes. Uniqueness now applies inside each
    # tier only, which is the real constraint (6 robots on the field at once).
    auto_lo, auto_hi = _auto_window(match, tracks)

    def _in_auto(tid):
        fr = tracks[tid]["frames"]
        return any(auto_lo <= f <= auto_hi for f in fr)

    # Within a tier, a team may be held by SEVERAL tracks as long as they do not
    # overlap in time. One team is one robot at any instant, but across the ~2:40
    # match that robot is split into dozens of fragments, so a flat "six teams,
    # six tracks" cap identifies at most 6 of ~160 tracks and every later fragment
    # goes unnamed no matter how clearly its bumper was read.
    def _overlaps(tid, spans):
        fr = tracks[tid]["frames"]
        lo, hi = min(fr), max(fr)
        return any(not (hi < a or lo > b) for a, b in spans)

    out = {}
    for tier in ("auto", "match"):
        used = defaultdict(list)  # team -> [(lo, hi), ...] already claimed
        tier_tids = [t for t in votes if t in tracks and _in_auto(t) == (tier == "auto")]
        ranked = sorted(tier_tids, key=lambda t: -sum(votes[t].values()))
        for tid in ranked:
            c = votes[tid]
            if not c:
                continue
            for team, n in c.most_common():
                # a team is free unless a track ALIVE AT THE SAME TIME already holds it
                if _overlaps(tid, used[team]):
                    continue
                total = sum(c.values())
                if n >= MIN_VOTES:
                    fr = tracks[tid]["frames"]
                    out[str(tid)] = {
                        "alliance": tracks[tid]["alliance"],
                        "team": team,
                        "confidence": round(n / total, 3),
                        "votes": {k: round(v, 2) for k, v in c.items()},
                        "source": "read",
                        "window": tier,
                    }
                    used[team].append((min(fr), max(fr)))
                break

    outdir = os.path.join(RESULTS, match)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "identity.json"), "w") as fh:
        json.dump({"match": match, "candidates": candidates, "tracks": out}, fh, indent=2)
    # the auto-tier count is the one that matters -- auto_paths.py only ever sees
    # tracks alive in the auto window, so a match-tier read is dead weight to it.
    in_auto = sorted({v["team"] for v in out.values() if v.get("window") == "auto"})
    n_auto_tracks = sum(1 for v in out.values() if v.get("window") == "auto")
    print(
        f"  {match}: identified {len(out)} tracks by READ "
        f"({n_auto_tracks} in AUTO, covering {len(in_auto)}/{len(candidates)} teams) "
        f"-> {in_auto}"
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match")
    args = p.parse_args()
    if args.match:
        run(args.match)
        return
    from ultralytics import YOLO
    import trajectories as tj

    model = YOLO(tj._newest_weights())
    for tr in sorted(glob.glob(os.path.join(RESULTS, "*", "trajectory.csv"))):
        run(os.path.basename(os.path.dirname(tr)), model)


if __name__ == "__main__":
    main()
