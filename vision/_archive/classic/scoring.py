"""
Scoring engine: figure out which robot scored which ball.

Two ideas do the work:

1. POSSESSION. Every frame, each visible ball is attributed to the nearest
   robot within a possession radius (or inside its bumper box). We remember the
   *last* robot that possessed each ball -- so if a robot shoots a ball into a
   goal, the ball flies free for a few frames but the last possessor is still
   the shooter.

2. SCORING. A ball "scores" when its track is DELETED (vanished for good) while
   its last known position is inside a goal zone -- i.e. it disappeared into the
   hub/goal. Credit goes to the ball's last possessor.

This is a heuristic, not ground truth -- it has no depth and no rules engine --
but on reasonably framed footage it turns raw pixels into a per-robot score
tally, which is the whole point.
"""

from collections import defaultdict


class ScoringEngine:
    def __init__(self, zones, possession_radius, possession_bbox_margin):
        self.zones = zones  # list[GoalZone]
        self.possession_radius = possession_radius
        self.possession_bbox_margin = possession_bbox_margin

        self.ball_possessor = {}  # ball_id -> (alliance, robot_id)
        self.scored_ball_ids = set()  # balls already counted (don't double count)
        self.events = []  # list of scoring-event dicts
        self.robot_scores = defaultdict(int)  # (alliance, robot_id) -> count
        self.unattributed = 0  # scored balls with no known possessor

    # -- possession -----------------------------------------------------------
    def update_possession(self, ball_tracks, robots_by_alliance):
        """robots_by_alliance: {'red': live_tracks_dict, 'blue': live_tracks_dict}.
        Only *visible* balls update possession (a coasting/occluded ball keeps
        whatever possessor it had)."""
        for bid, ball in ball_tracks.items():
            if not ball.visible or bid in self.scored_ball_ids:
                continue
            cx, cy = ball.centroid
            best = None
            best_dist = self.possession_radius
            for alliance, robots in robots_by_alliance.items():
                for rid, robot in robots.items():
                    rx, ry = robot.centroid
                    dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
                    inside = _inside_bbox(cx, cy, robot.bbox, self.possession_bbox_margin)
                    # inside-the-bumper-box wins outright; otherwise nearest.
                    if inside:
                        dist = 0.0
                    if dist < best_dist:
                        best_dist = dist
                        best = (alliance, rid)
            if best is not None:
                self.ball_possessor[bid] = best

    # -- scoring --------------------------------------------------------------
    def check_scores(self, deleted_ball_tracks, frame_idx, time_sec):
        """A deleted ball whose last position sits in a goal zone is a score."""
        new_events = []
        for ball in deleted_ball_tracks:
            if ball.id in self.scored_ball_ids:
                continue
            cx, cy = ball.centroid
            zone = self._zone_at(cx, cy)
            if zone is None:
                continue  # ball left frame / was set down mid-field -> not a score

            possessor = self.ball_possessor.get(ball.id)
            if possessor is not None:
                alliance, rid = possessor
            elif zone.alliance != "shared":
                # no possession record but it landed in a side goal -> credit
                # that alliance, unknown robot.
                alliance, rid = zone.alliance, None
            else:
                alliance, rid = None, None

            if rid is not None:
                self.robot_scores[(alliance, rid)] += 1
            else:
                self.unattributed += 1

            event = {
                "frame": frame_idx,
                "time_sec": round(time_sec, 3),
                "ball_id": ball.id,
                "zone": zone.name,
                "credited_alliance": alliance if alliance else "unknown",
                "credited_robot": rid if rid is not None else "unknown",
                "x": round(cx, 1),
                "y": round(cy, 1),
            }
            self.events.append(event)
            new_events.append(event)
            self.scored_ball_ids.add(ball.id)
        return new_events

    # -- tallies --------------------------------------------------------------
    def alliance_total(self, alliance):
        return sum(n for (a, _), n in self.robot_scores.items() if a == alliance)

    def robot_total(self, alliance, rid):
        return self.robot_scores.get((alliance, rid), 0)

    def _zone_at(self, cx, cy):
        for z in self.zones:
            if z.contains(cx, cy):
                return z
        return None


def _inside_bbox(cx, cy, bbox, margin):
    x, y, w, h = bbox
    return (x - margin) <= cx <= (x + w + margin) and (y - margin) <= cy <= (y + h + margin)
