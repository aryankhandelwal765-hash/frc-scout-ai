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

MAX_GAP_SEC = 6.0  # longest blind gap we will bridge
BASE_GATE_FRAC = 0.045  # gate at zero gap, as a fraction of crop width
GATE_GROWTH_FRAC = 0.020  # extra gate allowed per second of gap
MAX_SPEED_FRAC = 0.45  # cap on extrapolated speed, crop widths per second
FIT_SAMPLES = 6  # samples used to estimate a fragment's end velocity


def crop_width(match):
    """Field-crop width in px for this match's video, resolution-aware."""
    import trajectories as tj

    cap = cv2.VideoCapture(os.path.join(VIDEOS, f"{match}.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 712
    cap.release()
    x1, _y1, x2, _y2 = tj.field_crop((h, w, 3))
    return float(x2 - x1)


DUP_DIST_FRAC = 0.025  # two tracks closer than this (crop widths) at the same frame are one robot
DUP_MIN_SHARED = 0.5  # share of the shorter track's frames that must coincide with the longer


def dedupe(tracks, alliance_of, cw):
    """Merge duplicate tracks: same alliance, same frames, same place = one robot.

    A detector that boxes one robot twice (or ByteTrack spawning a second id on it) gives
    two tracks that run side by side. Stitching then treats them as two robots and they
    steal each other's successors. Fold the shorter into the longer: its rows only fill
    frames the longer one lacks.

    returns (tracks without the absorbed ids, {absorbed_tid: kept_tid})
    """
    dist = DUP_DIST_FRAC * cw
    by_len = sorted(tracks, key=lambda t: len(tracks[t]), reverse=True)
    rows = {t: {r[0]: r for r in tracks[t]} for t in by_len}
    merged, kept = {}, []
    for t in by_len:
        if len(tracks[t]) < 3:
            continue
        host = None
        for k in kept:
            if alliance_of.get(k) != alliance_of.get(t):
                continue
            shared = [f for f in rows[t] if f in rows[k]]
            if len(shared) < DUP_MIN_SHARED * len(rows[t]):
                continue
            close = sum(
                1
                for f in shared
                if ((rows[t][f][3] - rows[k][f][3]) ** 2 + (rows[t][f][4] - rows[k][f][4]) ** 2)
                ** 0.5
                <= dist
            )
            if close >= DUP_MIN_SHARED * len(rows[t]):
                host = k
                break
        if host is None:
            kept.append(t)
        else:
            merged[t] = host
            for f, r in rows[t].items():
                rows[host].setdefault(f, r)
    out = {t: sorted(rows[t].values()) for t in kept}
    for t in tracks:  # tiny tracks pass through untouched
        if t not in out and t not in merged:
            out[t] = tracks[t]
    return out, merged


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


def chains(match, tracks, alliance_of, fps=60.0, min_frames=3, window=None, dedup=False):
    """Group track ids into per-robot chains.

    tracks       : tid -> [(frame, t, alliance, cx, cy, conf), ...]
    alliance_of  : tid -> "blue"|"red"
    window       : optional (lo, hi) frame bounds to restrict to
    returns      : list of [tid, ...], each list ordered in time
    """
    cw = crop_width(match)
    if dedup:
        tracks, _gone = dedupe(tracks, alliance_of, cw)
    base = BASE_GATE_FRAC * cw
    growth = GATE_GROWTH_FRAC * cw
    max_speed = MAX_SPEED_FRAC * cw / fps  # px per frame
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
                continue  # must be strictly after, and not too far
            sp = min((avx**2 + avy**2) ** 0.5, max_speed)
            norm = (avx**2 + avy**2) ** 0.5 or 1.0
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
            continue  # one successor, one predecessor
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
    for tid in info:  # any leftover (cycle-broken) fragment
        if tid not in done:
            out.append([tid])
            done.add(tid)
    return out


SLOT_ANCHOR_SEC = 1.5  # a fragment this long may claim an empty slot; shorter ones only join one
SLOT_OVERLAP_SEC = 0.5  # a new id may start this long before the old one is dropped
ALLIANCE_SIZE = 3


def slot_chains(match, tracks, alliance_of, fps=60.0, min_frames=3, window=None, reads=None):
    """Chains via persistent per-alliance slots ("ghost" robots).

    An alliance fields exactly ALLIANCE_SIZE robots, so instead of asking "which fragment
    follows this one" we keep ALLIANCE_SIZE slots alive for the whole match. A slot that
    lost its robot stays where it vanished (ghost) and the next fragment that starts
    reachable from it takes it over, however long the blind gap was. A fragment that
    appears while every slot is busy is a duplicate or junk, so it is attached to the
    nearest busy slot instead of founding a fourth robot.

    reads: optional tid -> team (bumper reads). A slot belongs to the first team read on it;
    a fragment reading a different team can never join it, and one reading the slot's team
    is pulled to it. Same return shape as chains().
    """
    cw = crop_width(match)
    max_speed = MAX_SPEED_FRAC * cw / fps
    base = BASE_GATE_FRAC * cw
    overlap = SLOT_OVERLAP_SEC * fps
    anchor = SLOT_ANCHOR_SEC * fps

    info = {}
    for tid, rows in tracks.items():
        r = [x for x in rows if not window or window[0] <= x[0] <= window[1]]
        if len(r) >= min_frames:
            info[tid] = _endpoints(r)

    out = []
    for ally in sorted({alliance_of.get(t) for t in info}):
        frags = sorted((t for t in info if alliance_of.get(t) == ally), key=lambda t: info[t][0])
        slots = []  # each: {"chain": [tids], "end": f1, "pos": (x, y), "vel": (vx, vy)}
        junk = []
        for tid in frags:
            f0, f1, start, end, vx, vy = info[tid]
            best, best_cost = None, None
            team = (reads or {}).get(tid)
            for sl in slots:
                if team and sl["team"] and sl["team"] != team:
                    continue  # the bumper says a different robot
                gap = f0 - sl["end"]
                if gap < -overlap:
                    continue  # slot still held by a concurrent fragment
                gap = max(gap, 0)
                sp = min((sl["vel"][0] ** 2 + sl["vel"][1] ** 2) ** 0.5, max_speed)
                norm = (sl["vel"][0] ** 2 + sl["vel"][1] ** 2) ** 0.5 or 1.0
                # the ghost drifts at its last velocity for a few seconds, then holds still
                drift = min(gap, 3 * fps)
                px = sl["pos"][0] + sl["vel"][0] / norm * sp * drift
                py = sl["pos"][1] + sl["vel"][1] / norm * sp * drift
                d = ((start[0] - px) ** 2 + (start[1] - py) ** 2) ** 0.5
                reach = base + max_speed * gap  # farthest it could physically have gone
                if d > reach:
                    continue
                cost = d / (base + max_speed * gap * 0.5)
                if team and sl["team"] == team:
                    cost -= 100.0  # a matching read beats any geometry
                if best_cost is None or cost < best_cost:
                    best, best_cost = sl, cost
            if best is None and len(slots) < ALLIANCE_SIZE and f1 - f0 >= anchor:
                best = {"chain": [], "end": f0, "pos": start, "vel": (0.0, 0.0), "team": None}
                slots.append(best)
            if best is None:
                junk.append(tid)
                continue
            best["chain"].append(tid)
            if team and not best["team"]:
                best["team"] = team
            if f1 >= best["end"]:
                best["end"], best["pos"], best["vel"] = f1, end, (vx, vy)
        out.extend(sl["chain"] for sl in slots)
        out.extend([t] for t in junk)
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
    scored.sort(reverse=True)  # strongest evidence claims first

    taken = defaultdict(list)  # team -> [(lo, hi), ...] already claimed
    for _n, _len, i, team in scored:
        if not spans:  # no track data: fall back to global rule
            if team in taken:
                continue
            taken[team].append((0, 0))
        else:
            lo, hi = spans.get(i, (None, None))
            if lo is None:
                continue
            if any(not (hi < a or lo > b) for a, b in taken[team]):
                continue  # a concurrent chain already holds it
            taken[team].append((lo, hi))
        per_chain[i] = team
        for tid in chain_list[i]:
            per_tid[tid] = team
    return per_tid, per_chain
