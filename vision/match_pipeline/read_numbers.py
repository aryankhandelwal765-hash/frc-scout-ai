"""
Read BUMPER TEAM NUMBERS off robots -- the direct-evidence identity source.

At 1080p the bumper digits are legible in both the top view and the low reverse
views. Crucially this is NOT open-ended OCR: TBA tells us the six teams playing, so
a read only has to pick the nearest of six candidates. A partial/noisy read ("53_8")
still resolves to 5348 with high confidence, which is what makes identity robust.

Per detected robot:
  1. crop the bumper band at FULL resolution
  2. CLAHE-lift it (dim arena), upscale, binarise
  3. OCR digits (a few threshold variants -- take the best-scoring)
  4. score each candidate team by string similarity; accept if the best clearly
     beats the runner-up

  read_bumper(crop_bgr, box, candidates) -> (team, confidence) or (None, 0.0)
"""

import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Scene-text OCR. Tesseract is built for documents and returns NOTHING on these
# small, angled, dim bumper digits (measured: 0/107). EasyOCR is a scene-text model
# and does read them, if imperfectly ("5345" for 5348) -- which is fine, because we
# only have to pick the nearest of the six teams TBA says are in the match.
_READER = None


def reader():
    global _READER
    if _READER is None:
        import ssl, certifi, warnings

        warnings.filterwarnings("ignore")
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(
            cafile=certifi.where()
        )
        import easyocr

        _READER = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _READER


BUMPER_TOP = 0.62  # bumper band starts this far down the robot box
UPSCALE = 6  # digits are small; upscale before OCR
MIN_MARGIN = 0.15  # best candidate must beat 2nd by this much
MIN_SCORE = 0.55  # and clear this absolute similarity


def _variants(gray):
    """A few binarisations -- bumper digits are white-on-colour, and lighting
    varies a lot, so no single threshold wins everywhere."""
    out = []
    _, o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(o)
    out.append(cv2.bitwise_not(o))
    out.append(
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    )
    return out


def _ocr_digits(img):
    """All digit strings EasyOCR finds, best-confidence first."""
    try:
        res = reader().readtext(img, allowlist="0123456789")
    except Exception:
        return []
    out = [(re.sub(r"\D", "", t), float(c)) for _, t, c in res]
    out = [(t, c) for t, c in out if t]
    out.sort(key=lambda tc: -tc[1])
    return out


def _similarity(read, team):
    """How well a (possibly partial) OCR string matches a candidate team number.
    Exact = 1.0; otherwise reward matching length + shared digits in position."""
    if not read:
        return 0.0
    if read == team:
        return 1.0
    s = 0.0
    if len(read) == len(team):
        same = sum(1 for a, b in zip(read, team) if a == b)
        s = same / len(team)
    elif read in team:  # partial read, e.g. "534" of "5348"
        s = 0.8 * len(read) / len(team)
    else:
        # digit-set overlap as a weak fallback
        common = len(set(read) & set(team))
        s = 0.4 * common / max(len(set(team)), 1)
    return s


def bumper_crop(img, box, top_frac=BUMPER_TOP):
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    by1 = y1 + int(h * top_frac)
    return img[max(0, by1) : max(0, y2), max(0, x1) : max(0, x2)]


def read_bumper(img, box, candidates, debug=False):
    """img: full-res BGR frame. box: robot box in the SAME coords.
    candidates: list of team-number strings (the 6 teams in this match).
    Returns (team, confidence)."""
    crop = bumper_crop(img, box)
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 8:
        return None, 0.0
    big = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    gray = cv2.cvtColor(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)

    reads = []
    for cand_img in (big, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)):
        reads += [t for t, c in _ocr_digits(cand_img)]
    if not reads:
        return None, 0.0

    best = {}
    for team in candidates:
        best[team] = max(_similarity(r, team) for r in reads)
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    top_team, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if debug:
        print("   reads:", reads, "->", ranked[:3])
    if top_score >= MIN_SCORE and (top_score - second) >= MIN_MARGIN:
        return top_team, round(float(top_score), 3)
    return None, round(float(top_score), 3)


def match_candidates(match, truth_dir=None):
    """The six team numbers playing this match, from TBA truth."""
    import json

    truth_dir = truth_dir or os.path.join(HERE, "truth")
    p = os.path.join(truth_dir, f"{match}.json")
    if not os.path.exists(p):
        return []
    t = json.load(open(p))
    out = []
    for side in ("blue_teams", "red_teams"):
        for k in t.get(side) or []:
            out.append(str(k).replace("frc", ""))
    return out


# --- side (reverse) views -------------------------------------------------
# Each low reverse camera looks at ONE alliance's half, so the candidate set is
# that alliance's 3 teams, not all 6. Halving the candidates roughly halves the
# chance of a near-miss snapping to the wrong number.
#
# And because we already know exactly which strings to look for, the side views
# need no robot detector at all: OCR the whole frame and keep any digit blob that
# snaps to one of the 3. A hit gives identity AND position in one step.


def alliance_candidates(match, alliance, truth_dir=None):
    """The 3 team numbers on one alliance ('blue'|'red')."""
    import json

    truth_dir = truth_dir or os.path.join(HERE, "truth")
    p = os.path.join(truth_dir, f"{match}.json")
    if not os.path.exists(p):
        return []
    t = json.load(open(p))
    return [str(k).replace("frc", "") for k in (t.get(f"{alliance}_teams") or [])]


def scan_side(frame, candidates, min_conf=0.10, upscale=2, favour=(), bonus=0.10):
    """OCR a whole reverse-view frame; return [(team, (x1,y1,x2,y2), score, raw)]
    for every digit blob that snaps to one of `candidates`.

    `favour` = the teams this camera mostly looks at (its near alliance). Measured
    on 2026hop_qm27: revL is blue-leaning 4:1, revR red-leaning 3:2 -- a LEAN, not a
    rule, because each low camera also sees across the field. Hard-filtering to one
    alliance threw away 3 of 10 valid reads, so the side is applied as a small
    tie-break bonus instead of a filter."""
    img = frame
    if upscale != 1:
        img = cv2.resize(frame, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    try:
        res = reader().readtext(img, allowlist="0123456789")
    except Exception:
        return []
    out = []
    for bb, txt, conf in res:
        raw = re.sub(r"\D", "", txt)
        if not raw or conf < min_conf:
            continue
        scored = sorted(
            ((_similarity(raw, c) + (bonus if c in favour else 0.0), c) for c in candidates),
            reverse=True,
        )
        best, team = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best >= MIN_SCORE and (best - second) >= MIN_MARGIN:
            xs = [p[0] / upscale for p in bb]
            ys = [p[1] / upscale for p in bb]
            out.append((team, (min(xs), min(ys), max(xs), max(ys)), round(float(best), 3), raw))
    return out
