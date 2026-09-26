"""
Render each robot's AUTO zone sequence as a field PNG.

Visited zones are highlighted with a green→red gradient (start=green, end=red).
Numbered circles mark visit order; arrows connect consecutive zones.

Outputs:
  results/<match>/zone_path_<station>.png   -- per robot per match
  zone_paths/<team>.png                     -- team's zone sequences across matches (grid)

Run:
  python render_zone_paths.py                         # all matches
  python render_zone_paths.py --match 2026hop_qm27    # one match
  python render_zone_paths.py --force                 # redo existing
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import topdown as td
import zones

RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(HERE, "zone_paths")


def _step_color(i, n):
    """BGR color gradient: green (step 0) → yellow → red (last step)."""
    t = i / max(n - 1, 1)
    r = int(40 + 210 * t)
    g = int(220 - 180 * t)
    b = 40
    return (b, g, r)


def _zone_center_canvas(zone_name):
    """Field-inch zone center → canvas pixel (cx, cy)."""
    x0, y0, x1, y1 = zones.ZONES[zone_name]
    cx = int((x0 + x1) / 2 * (td.FX1 - td.FX0) / zones.FIELD_LEN + td.FX0)
    cy = int((y0 + y1) / 2 * (td.FY1 - td.FY0) / zones.FIELD_WID + td.FY0)
    return cx, cy


def _zone_rect_canvas(zone_name):
    """Field-inch zone rect → canvas pixel (px0, py0, px1, py1)."""
    x0, y0, x1, y1 = zones.ZONES[zone_name]
    px0 = int(x0 * (td.FX1 - td.FX0) / zones.FIELD_LEN + td.FX0)
    py0 = int(y0 * (td.FY1 - td.FY0) / zones.FIELD_WID + td.FY0)
    px1 = int(x1 * (td.FX1 - td.FX0) / zones.FIELD_LEN + td.FX0)
    py1 = int(y1 * (td.FY1 - td.FY0) / zones.FIELD_WID + td.FY0)
    return px0, py0, px1, py1


def render_zone_sequence(zone_seq, title="", alliance="blue"):
    """Return a field PNG (numpy BGR) with the zone visit sequence visualised.

    Usable as a standalone image AND as context for human scouts reviewing paths.
    zone_seq: ordered list of zone names (from zones.zone_sequence())
    """
    img = td.draw_field(blue_left=(alliance != "red"))

    valid = [z for z in zone_seq if z in zones.ZONES]
    if not valid:
        if title:
            cv2.putText(img, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 2)
        cv2.putText(
            img,
            "No zones detected",
            (16, td.CH // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (140, 140, 140),
            2,
        )
        return img

    n = len(valid)

    # Semi-transparent zone fills
    overlay = img.copy()
    for i, zone_name in enumerate(valid):
        px0, py0, px1, py1 = _zone_rect_canvas(zone_name)
        cv2.rectangle(overlay, (px0, py0), (px1, py1), _step_color(i, n), -1)
    cv2.addWeighted(overlay, 0.32, img, 0.68, 0, img)

    # Zone border outlines
    for i, zone_name in enumerate(valid):
        px0, py0, px1, py1 = _zone_rect_canvas(zone_name)
        cv2.rectangle(img, (px0, py0), (px1, py1), _step_color(i, n), 2)

    # Arrows between consecutive unique zone centers
    centers = [_zone_center_canvas(z) for z in valid]
    for i in range(len(centers) - 1):
        a, b = centers[i], centers[i + 1]
        if a == b:
            continue
        col = _step_color(i, n)
        cv2.arrowedLine(img, a, b, (255, 255, 255), 5, cv2.LINE_AA, tipLength=0.12)
        cv2.arrowedLine(img, a, b, col, 3, cv2.LINE_AA, tipLength=0.12)

    # Numbered circles at each center
    for i, (zone_name, center) in enumerate(zip(valid, centers)):
        col = _step_color(i, n)
        cv2.circle(img, center, 20, (255, 255, 255), -1)
        cv2.circle(img, center, 20, col, 3)
        num = str(i + 1)
        tw, th = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
        cv2.putText(
            img,
            num,
            (center[0] - tw // 2, center[1] + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
        )

    # Short zone labels offset from center
    for i, (zone_name, center) in enumerate(zip(valid, centers)):
        short = (
            zone_name.replace("Neutral ", "N.")
            .replace("Blue ", "B.")
            .replace("Red ", "R.")
            .replace(" Trench", " Tr")
            .replace(" Bump", " Bmp")
        )
        lx, ly = center[0] + 24, center[1] + 6
        cv2.putText(
            img, short, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 4, cv2.LINE_AA
        )
        cv2.putText(
            img, short, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1, cv2.LINE_AA
        )

    # Title
    if title:
        cv2.putText(
            img, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 6, cv2.LINE_AA
        )
        cv2.putText(
            img, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA
        )

    # Legend bottom
    legend = "  ".join(f"{i + 1}={z}" for i, z in enumerate(valid))
    cv2.putText(
        img, legend[:120], (16, td.CH - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1
    )

    return img


def render_match_zone_paths(match, force=False):
    """Render zone_path_<station>.png for every robot in one match."""
    interp_path = os.path.join(RESULTS, match, "auto_interp.json")
    if not os.path.exists(interp_path):
        return 0
    try:
        data = json.load(open(interp_path))
    except Exception:
        return 0

    n = 0
    for station, info in data.items():
        if "error" in info or not info.get("zone_seq"):
            continue
        out = os.path.join(RESULTS, match, f"zone_path_{station}.png")
        if not force and os.path.exists(out):
            continue
        team = info.get("team", station)
        team_label = team.replace("frc", "#") if str(team).startswith("frc") else str(team)
        name = info.get("name", "")
        title = f"{team_label}  {match}  {name}"
        img = render_zone_sequence(
            info["zone_seq"], title=title, alliance=info.get("alliance", "blue")
        )
        cv2.imwrite(out, img)
        n += 1
    return n


def render_team_zone_grid(team, force=False):
    """One PNG per team showing all their zone-sequence renders in a grid."""
    os.makedirs(OUTDIR, exist_ok=True)
    team_num = team.replace("frc", "")
    out = os.path.join(OUTDIR, f"{team_num}.png")
    if not force and os.path.exists(out):
        return out

    frames = []
    for interp_path in sorted(glob.glob(os.path.join(RESULTS, "*", "auto_interp.json"))):
        match = os.path.basename(os.path.dirname(interp_path))
        try:
            data = json.load(open(interp_path))
        except Exception:
            continue
        for station, info in data.items():
            if info.get("team") != team or not info.get("zone_seq"):
                continue
            name = info.get("name", "")
            title = f"{match} {station} {name}"
            img = render_zone_sequence(
                info["zone_seq"], title=title, alliance=info.get("alliance", "blue")
            )
            # Thumbnail: scale to 1/3
            h, w = img.shape[:2]
            thumb = cv2.resize(img, (w // 3, h // 3))
            frames.append(thumb)

    if not frames:
        return None

    # Stack in rows of 3
    cols = 3
    rows = (len(frames) + cols - 1) // cols
    # Pad to fill last row
    while len(frames) < rows * cols:
        frames.append(np.full_like(frames[0], 240))
    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.hstack(frames[r * cols : (r + 1) * cols]))
    grid = np.vstack(grid_rows)
    cv2.imwrite(out, grid)
    return out


def render_team_zone_aggregate(team, force=False, start_side=None):
    """Single field PNG: zone visit frequency + transition arrows across ALL matches.

    Zone fill opacity ∝ visit count. Arrow thickness ∝ transition frequency.
    start_side: None = all, "left" | "centre" | "right" = filter to that side only.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    team_num = team.replace("frc", "")
    suffix = f"_{start_side}" if start_side else ""
    out = os.path.join(OUTDIR, f"{team_num}_aggregate{suffix}.png")
    if not force and os.path.exists(out):
        return out

    from collections import Counter

    zone_counts: Counter = Counter()
    transition_counts: Counter = Counter()
    n_matches = 0

    for interp_path in sorted(glob.glob(os.path.join(RESULTS, "*", "auto_interp.json"))):
        try:
            data = json.load(open(interp_path))
        except Exception:
            continue
        for station, info in data.items():
            if info.get("team") != team or not info.get("zone_seq"):
                continue
            info_side = info.get("start_side", "")
            if info_side == "center":
                info_side = "centre"
            if start_side and info_side != start_side:
                continue
            seq = [z for z in info["zone_seq"] if z in zones.ZONES]
            for z in seq:
                zone_counts[z] += 1
            for a, b in zip(seq, seq[1:]):
                if a != b:
                    transition_counts[(a, b)] += 1
            n_matches += 1

    if not zone_counts:
        return None

    img = td.draw_field(blue_left=True)
    max_visits = max(zone_counts.values())
    max_trans = max(transition_counts.values()) if transition_counts else 1

    # Semi-transparent zone fills scaled by visit frequency
    overlay = img.copy()
    for zone_name, count in zone_counts.items():
        intensity = count / max_visits  # 0..1
        px0, py0, px1, py1 = _zone_rect_canvas(zone_name)
        # Blue tint scaled by intensity
        col = (
            int(40 + 180 * intensity),
            int(60 + 140 * (1 - intensity)),
            int(20 + 220 * intensity),
        )
        cv2.rectangle(overlay, (px0, py0), (px1, py1), col, -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # Zone borders + visit count labels
    for zone_name, count in zone_counts.items():
        px0, py0, px1, py1 = _zone_rect_canvas(zone_name)
        intensity = count / max_visits
        col = (
            int(40 + 180 * intensity),
            int(60 + 140 * (1 - intensity)),
            int(20 + 220 * intensity),
        )
        cv2.rectangle(img, (px0, py0), (px1, py1), col, 2)

        cx, cy = _zone_center_canvas(zone_name)
        lbl = str(count)
        tw, th = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.circle(img, (cx, cy), 18, (255, 255, 255), -1)
        cv2.circle(img, (cx, cy), 18, col, 3)
        cv2.putText(
            img, lbl, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2
        )

        short = (
            zone_name.replace("Neutral ", "N.")
            .replace("Blue ", "B.")
            .replace("Red ", "R.")
            .replace(" Trench", " Tr")
            .replace(" Bump", " Bmp")
        )
        cv2.putText(
            img,
            short,
            (cx + 22, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            short,
            (cx + 22, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    # Transition arrows: thickness ∝ frequency
    for (a, b), count in sorted(transition_counts.items(), key=lambda x: x[1]):
        ca, cb = _zone_center_canvas(a), _zone_center_canvas(b)
        if ca == cb:
            continue
        t = count / max_trans
        thick = max(1, int(1 + 7 * t))
        alpha_col = (int(220 * t), int(180 * t), int(220 * t))
        cv2.arrowedLine(img, ca, cb, (255, 255, 255), thick + 3, cv2.LINE_AA, tipLength=0.1)
        cv2.arrowedLine(img, ca, cb, alpha_col, thick, cv2.LINE_AA, tipLength=0.1)
        # Count on thicker arrows
        if count >= 2:
            mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
            cv2.putText(
                img,
                str(count),
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                img, str(count), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA
            )

    team_label = team.replace("frc", "#") if str(team).startswith("frc") else team
    side_label = f"  [{start_side.upper()} START]" if start_side else ""
    title = f"{team_label}  AUTO ZONE AGGREGATE{side_label}  ({n_matches} matches)"
    cv2.putText(
        img, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 6, cv2.LINE_AA
    )
    cv2.putText(img, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        "circle = visit count   arrow thickness = transition frequency",
        (16, td.CH - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (80, 80, 80),
        1,
    )

    cv2.imwrite(out, img)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--match", default="", help="single match key; empty = all")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    if a.match:
        n = render_match_zone_paths(a.match, a.force)
        print(f"{a.match}: {n} zone path PNGs")
        return

    total = 0
    interp_paths = sorted(glob.glob(os.path.join(RESULTS, "*", "auto_interp.json")))
    for ip in interp_paths:
        match = os.path.basename(os.path.dirname(ip))
        n = render_match_zone_paths(match, a.force)
        if n:
            print(f"  {match}: {n} PNGs")
        total += n
    print(f"\nTotal: {total} zone path PNGs in results/*/zone_path_*.png")

    # Per-team grids
    teams = set()
    for ip in interp_paths:
        try:
            for info in json.load(open(ip)).values():
                t = info.get("team")
                if t:
                    teams.add(t)
        except Exception:
            pass
    kept = 0
    for team in sorted(teams):
        if render_team_zone_grid(team, a.force):
            kept += 1
    print(f"Per-team zone grids: {kept} -> {OUTDIR}/")


if __name__ == "__main__":
    main()
