"""
Stitch ByteTrack fragments back into ONE chain per physical robot.

The problem this exists to solve, measured on 2026hop_qm100: red robot #1706 is
read correctly off its bumper on track 19 (frames 182-274), track 19 then dies, the
robot is re-acquired as new ids, and the auto book labels those later fragments by
rank order -- so the same robot is called 1706 for two seconds and 3035 afterwards.
The read was right. It just did not survive the id change.

So: link fragment B to fragment A when B starts, shortly after A ended, near where A
was heading. Chain them, and a read anywhere on the chain locks the whole chain.

Gating is VELOCITY-AWARE. A robot that was moving keeps moving during the gap, so
comparing B's start against A's last position alone rejects real links; we compare
against A's position extrapolated forward across the gap, and widen the gate with
the gap length because the extrapolation gets worse the longer the robot is unseen.

All distances are FRACTIONS of the field-crop width -- no pixel literals, so this
survives a resolution change.
"""

import os
from collections import defaultdict

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEOS = os.path.join(HERE, "videos")

MAX_GAP_SEC = 6.0        # longest blind gap we will bridge
BASE_GATE_FRAC = 0.045   # gate at zero gap, as a fraction of crop width
GATE_GROWTH_FRAC = 0.020 # extra gate allowed per second of gap
MAX_SPEED_FRAC = 0.45    # cap on extrapolated speed, crop widths per second
FIT_SAMPLES = 6          # samples used to estimate a fragment's end velocity


def crop_width(match):
    """Field-crop width in px for this match's video, resolution-aware."""
    import trajectories as tj
    cap = cv2.VideoCapture(os.path.join(VIDEOS, f"{match}.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 712
    cap.release()
    x1, _y1, x2, _y2 = tj.field_crop((h, w, 3))
    return float(x2 - x1)


def _endpoints(rows):
    """(first_frame, last_frame, start_xy, end_xy, vx, vy) for one fragment.

    Velocity is per FRAME, fitted over the last FIT_SAMPLES samples.
    """
    rows = sorted(rows)
    f0, f1 = rows[0][0], rows[-1][0]
    sx, sy = rows[0][3], rows[0][4]
    ex, ey = rows[-1][3], rows[-1][4]
    tail = rows[-FIT_SAMPLES:] if len(rows) >= 2 else rows
    df = tail[-1][0] - tail[0][0]
    if df > 0:
        vx = (tail[-1][3] - tail[0][3]) / df
        vy = (tail[-1][4] - tail[0][4]) / df
    else:
        vx = vy = 0.0
    return f0, f1, (sx, sy), (ex, ey), vx, vy


def chains(match, tracks, alliance_of, fps=60.0, min_frames=3, window=None):
    """Group track ids into per-robot chains.

    tracks       : tid -> [(frame, t, alliance, cx, cy, conf), ...]
    alliance_of  : tid -> "blue"|"red"
    window       : optional (lo, hi) frame bounds to restrict to
    returns      : list of [tid, ...], each list ordered in time
    """
    cw = crop_width(match)
    base = BASE_GATE_FRAC * cw
    growth = GATE_GROWTH_FRAC * cw
    max_speed = MAX_SPEED_FRAC * cw / fps          # px per frame
    max_gap = MAX_GAP_SEC * fps

    info = {}
    for tid, rows in tracks.items():
        r = rows
        if window:
            r = [x for x in rows if window[0] <= x[0] <= window[1]]
        if len(r) < min_frames:
            continue
        info[tid] = _endpoints(r)

    # candidate links A -> B, cheapest first
    cand = []
    for a, (_af0, af1, _as, ae, avx, avy) in info.items():
        for b, (bf0, _bf1, bs, _be, _bvx, _bvy) in info.items():
            if a == b or alliance_of.get(a) != alliance_of.get(b):
                continue
            gap = bf0 - af1
            if gap <= 0 or gap > max_gap:
                continue                    # must be strictly after, and not too far
            sp = min((avx ** 2 + avy ** 2) ** 0.5, max_speed)
            norm = (avx ** 2 + avy ** 2) ** 0.5 or 1.0
            px = ae[0] + (avx / norm) * sp * gap
            py = ae[1] + (avy / norm) * sp * gap
            d = ((bs[0] - px) ** 2 + (bs[1] - py) ** 2) ** 0.5
            gate = base + growth * (gap / fps)
            if d <= gate:
                cand.append((d / gate, gap, a, b))

    cand.sort()
    nxt, prv = {}, {}
    for _cost, _gap, a, b in cand:
        if a in nxt or b in prv:
            continue                        # one successor, one predecessor
        # no cycles: walk forward from b and refuse if we reach a
        seen, cur = set(), b
        while cur in nxt and cur not in seen:
            seen.add(cur)
            cur = nxt[cur]
        if cur == a:
            continue
        nxt[a], prv[b] = b, a

    out, done = [], set()
    for tid in info:
        if tid in prv or tid in done:
            continue
        chain, cur = [], tid
        while cur is not None and cur not in done:
            chain.append(cur)
            done.add(cur)
            cur = nxt.get(cur)
        out.append(chain)
    for tid in info:                        # any leftover (cycle-broken) fragment
        if tid not in done:
            out.append([tid])
            done.add(tid)
    return out


def lock_identity(chain_list, read_id, tracks=None):
    """chain -> team, by majority read across the chain. A read ANYWHERE on a chain
    locks the whole chain -- that is the entire point.

    Uniqueness is enforced only between chains that OVERLAP IN TIME. One team is
    one robot at any instant, but across a whole match that robot is legitimately
    split into many chains, so a global "one team, one chain" rule silently starves
    every later chain of an identity -- which is why whole-match attribution sat at
    17% while the auto window did fine. Pass `tracks` to get the time-aware rule;
    without it the old global rule applies.

    Returns (tid -> team, chain_index -> team).
    """
    per_tid, per_chain = {}, {}
    spans = {}
    if tracks:
        for i, ch in enumerate(chain_list):
            fr = [r[0] for tid in ch for r in tracks.get(tid, [])]
            if fr:
                spans[i] = (min(fr), max(fr))

    scored = []
    for i, ch in enumerate(chain_list):
        votes = defaultdict(int)
        for tid in ch:
            t = read_id.get(tid)
            if t:
                votes[t] += 1
        if votes:
            team, n = max(votes.items(), key=lambda kv: kv[1])
            scored.append((n, len(ch), i, team))
    scored.sort(reverse=True)               # strongest evidence claims first

    taken = defaultdict(list)               # team -> [(lo, hi), ...] already claimed
    for _n, _len, i, team in scored:
        if not spans:                       # no track data: fall back to global rule
            if team in taken:
                continue
            taken[team].append((0, 0))
        else:
            lo, hi = spans.get(i, (None, None))
            if lo is None:
                continue
            if any(not (hi < a or lo > b) for a, b in taken[team]):
                continue                    # a concurrent chain already holds it
            taken[team].append((lo, hi))
        per_chain[i] = team
        for tid in chain_list[i]:
            per_tid[tid] = team
    return per_tid, per_chain
