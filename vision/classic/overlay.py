"""
Overlay: draw everything onto the annotated output frame.

Robots (alliance-colored boxes + ID + running score), balls (circles + ID),
possession lines (ball -> the robot holding it), goal zones (translucent), and a
scoreboard panel. A brief "SCORE!" flash fires when an event lands.
"""

import cv2

# BGR
RED = (0, 0, 255)
BLUE = (255, 0, 0)
BALL = (0, 215, 255)  # amber
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 220, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

ALLIANCE_BGR = {"red": RED, "blue": BLUE, "shared": (0, 200, 200)}


def draw_zones(frame, zones):
    overlay = frame.copy()
    for z in zones:
        x, y, w, h = z.rect
        color = ALLIANCE_BGR.get(z.alliance, GREEN)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"GOAL:{z.name}", (x + 4, y + 20), FONT, 0.55, WHITE, 2)
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)


def draw_robots(frame, robots, alliance, engine):
    color = ALLIANCE_BGR[alliance]
    tag = "R" if alliance == "red" else "B"
    for rid, t in robots.items():
        x, y, w, h = [int(v) for v in t.bbox]
        # Dim boxes that are only coasting on the grace period.
        thickness = 2 if t.visible else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        score = engine.robot_total(alliance, rid)
        label = f"{tag}{rid}  balls:{score}"
        cv2.putText(frame, label, (x, y - 8), FONT, 0.55, color, 2)


def draw_balls(frame, balls, engine):
    # With hundreds of balls on the field, per-ball ID labels are illegible
    # noise -- only label when there are few enough to read.
    live = [(bid, t) for bid, t in balls.items() if bid not in engine.scored_ball_ids]
    show_labels = len(live) <= 25
    for bid, t in live:
        cx, cy = int(t.centroid[0]), int(t.centroid[1])
        # Track objects don't carry a radius; derive one from the bbox.
        x, y, w, h = t.bbox
        radius = max(6, int(max(w, h) / 2))
        cv2.circle(frame, (cx, cy), radius, BALL, 2)
        if show_labels:
            cv2.putText(frame, f"ball{bid}", (cx - 14, cy - radius - 6), FONT, 0.45, BALL, 1)


def draw_possession(frame, balls, robots_by_alliance, engine):
    for bid, t in balls.items():
        if bid in engine.scored_ball_ids:
            continue
        owner = engine.ball_possessor.get(bid)
        if not owner:
            continue
        alliance, rid = owner
        robot = robots_by_alliance.get(alliance, {}).get(rid)
        if robot is None:
            continue
        bx, by = int(t.centroid[0]), int(t.centroid[1])
        rx, ry = int(robot.centroid[0]), int(robot.centroid[1])
        cv2.line(frame, (bx, by), (rx, ry), GREEN, 1)


def draw_scoreboard(frame, engine, frame_idx, time_sec):
    h = frame.shape[0]
    panel_w, panel_h = 260, 150
    x0, y0 = 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    red_total = engine.alliance_total("red")
    blue_total = engine.alliance_total("blue")
    cv2.putText(frame, "SCORING (balls)", (x0 + 10, y0 + 24), FONT, 0.6, WHITE, 2)
    cv2.putText(frame, f"RED  {red_total}", (x0 + 10, y0 + 52), FONT, 0.7, RED, 2)
    cv2.putText(frame, f"BLUE {blue_total}", (x0 + 130, y0 + 52), FONT, 0.7, BLUE, 2)

    # Per-robot breakdown (top scorers first).
    rows = sorted(engine.robot_scores.items(), key=lambda kv: -kv[1])[:4]
    yy = y0 + 78
    for (alliance, rid), n in rows:
        tag = "R" if alliance == "red" else "B"
        cv2.putText(frame, f"{tag}{rid}: {n}", (x0 + 10, yy), FONT, 0.5, ALLIANCE_BGR[alliance], 1)
        yy += 20

    cv2.putText(
        frame, f"t={time_sec:5.1f}s  f={frame_idx}", (x0 + 130, y0 + 78), FONT, 0.45, WHITE, 1
    )


def draw_score_flash(frame, event, alpha=1.0):
    """Big centered SCORE! banner right after an event."""
    h, w = frame.shape[:2]
    txt = "SCORE!"
    sub = f"{str(event['credited_alliance']).upper()} {event['credited_robot']}  (goal:{event['zone']})"
    color = ALLIANCE_BGR.get(event["credited_alliance"], GREEN)
    overlay = frame.copy()
    (tw, th), _ = cv2.getTextSize(txt, FONT, 2.2, 6)
    cx = (w - tw) // 2
    cv2.putText(overlay, txt, (cx, h // 2), FONT, 2.2, color, 6)
    (sw, _), _ = cv2.getTextSize(sub, FONT, 0.8, 2)
    cv2.putText(overlay, sub, ((w - sw) // 2, h // 2 + 40), FONT, 0.8, WHITE, 2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
