"""
Configuration: HSV color presets, tunable parameters, and goal-zone data types.

All the numbers you'll actually want to change live here (or are exposed as CLI
flags in track_match.py). Broadcast footage varies a lot -- expect to tune.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Bumper (alliance) colors -- used to detect ROBOTS.
# Red wraps hue 0/180 in OpenCV HSV, so it needs two ranges.
# ---------------------------------------------------------------------------
RED_BUMPER_RANGES = [
    ((0, 120, 70), (10, 255, 255)),
    ((170, 120, 70), (180, 255, 255)),
]
BLUE_BUMPER_RANGES = [
    ((100, 120, 70), (130, 255, 255)),
]

# ---------------------------------------------------------------------------
# Game-piece (BALL) color presets.
#
# Default is "yellow" (2020/21 Power Cells, 2017 fuel) because a yellow ball is
# color-distinct from red/blue bumpers -- that separation is what makes the
# "robot vs ball" split reliable with pure color CV. If your game uses red/blue
# cargo, the ball and bumper colors overlap and you'll lean much harder on the
# circularity filter (balls are round, bumpers are not).
# ---------------------------------------------------------------------------
BALL_COLOR_PRESETS = {
    # Widened for real arena footage: balls are dimmer + more yellow-GREEN than
    # pure yellow, so the sat/val floor is low and the hue reaches into green.
    "yellow": [((18, 55, 80), (45, 255, 255))],  # power cells / fuel / tennis-ball yellow
    "yellow_bright": [((20, 100, 100), (35, 255, 255))],  # tight range for clean/synthetic footage
    "orange": [((8, 120, 120), (20, 255, 255))],  # basketballs / orange game pieces
    "green": [((40, 70, 60), (85, 255, 255))],  # teal/green (e.g. algae)
    "red": RED_BUMPER_RANGES,  # red cargo (overlaps red bumpers!)
    "blue": BLUE_BUMPER_RANGES,  # blue cargo (overlaps blue bumpers!)
}
DEFAULT_BALL_COLOR = "yellow"


@dataclass
class GoalZone:
    """A rectangular scoring region in PIXEL coordinates (x, y, w, h).

    alliance: 'red', 'blue', or 'shared' -- which alliance earns points for a
    ball that ends up here. 'shared' credits whichever robot last possessed the
    ball regardless of side (useful for a single central hub / shared goal).
    """

    name: str
    alliance: str  # 'red' | 'blue' | 'shared'
    rect: Tuple[int, int, int, int]

    def contains(self, cx: float, cy: float) -> bool:
        x, y, w, h = self.rect
        return x <= cx <= x + w and y <= cy <= y + h


@dataclass
class Params:
    """Everything tunable in one place. CLI flags override these."""

    # detection
    robot_min_area: int = 800  # px; raise to kill crowd/banner noise
    ball_min_area: int = 6  # px; distant arena balls are tiny (~8-15px wide)
    ball_max_area: int = 900  # px; reject big color patches / merged clumps
    ball_min_circularity: float = 0.45  # 1.0 == perfect circle; broadcast balls aren't perfect
    ball_close_kernel: int = 2  # small CLOSE to solidify balls; NO open (open erases them)

    # tracking
    robot_max_distance: float = 90.0  # px a robot centroid can jump between frames
    ball_max_distance: float = 70.0  # px a ball centroid can jump between frames
    robot_grace_frames: int = 8  # keep a robot ID alive this many missed frames
    ball_grace_frames: int = 15  # balls get occluded during possession -> be patient

    # scoring
    possession_radius: float = 60.0  # px; ball within this of a robot -> "possessed"
    possession_bbox_margin: float = 20.0  # also count as possessed if inside robot bbox + margin

    morph_kernel: int = 5  # px; opening kernel to clean color-mask noise
