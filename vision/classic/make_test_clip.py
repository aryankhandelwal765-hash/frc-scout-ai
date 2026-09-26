"""
Generate a SYNTHETIC FRC-ish match clip to validate the vision pipeline without
needing real footage. Deterministic, so the expected scoring outcome is known:

    Red  R0 -> scores 2 balls (carries ball A, then ball C, into the red goal)
    Blue B0 -> scores 1 ball  (carries ball B into the blue goal)
    Ball D  -> rolls off-screen, never enters a goal -> NOT scored

Writes: test_clip.mp4 and test_zones.json.
Run the tracker on it with:
    python track_match.py --input test_clip.mp4 --zones test_zones.json --outdir out
"""

import json

import cv2
import numpy as np

W, H, FPS, FRAMES = 960, 540, 30, 360

# Goal zones (pixel rects) -- these go into test_zones.json AND are drawn as
# neutral dark boxes in the source so they don't trip color detection.
RED_GOAL = (20, 200, 120, 140)  # x, y, w, h  (left)
BLUE_GOAL = (820, 200, 120, 140)  # right

BG = (110, 110, 110)  # neutral gray field (low saturation)
GOAL_FILL = (55, 55, 55)  # neutral dark (won't detect as robot/ball)
RED_BGR = (0, 0, 220)
BLUE_BGR = (220, 0, 0)
BALL_BGR = (0, 255, 255)  # pure yellow


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def red_x(f):
    if f <= 130:
        return lerp(450, 150, f / 130)
    if f <= 200:
        return lerp(150, 450, (f - 130) / 70)
    if f <= 315:
        return lerp(450, 150, (f - 200) / 115)
    return 150


def blue_x(f):
    if f <= 170:
        return lerp(520, 800, f / 170)
    return 800


def draw_robot(img, cx, cy, color):
    x0, y0 = int(cx - 30), int(cy - 30)
    cv2.rectangle(img, (x0, y0), (x0 + 60, y0 + 60), color, -1)


def draw_ball(img, cx, cy):
    cv2.circle(img, (int(cx), int(cy)), 12, BALL_BGR, -1)


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("test_clip.mp4", fourcc, FPS, (W, H))

    for f in range(FRAMES):
        img = np.full((H, W, 3), BG, dtype=np.uint8)
        for gx, gy, gw, gh in (RED_GOAL, BLUE_GOAL):
            cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), GOAL_FILL, -1)

        rx, by = red_x(f), 260
        bx = blue_x(f)
        draw_robot(img, rx, 260, RED_BGR)
        draw_robot(img, bx, 260, BLUE_BGR)

        # Ball A: carried by red robot until it enters the red goal, then vanishes.
        if f <= 125:
            draw_ball(img, rx - 40, 260)
        # Ball B: carried by blue robot into the blue goal, then vanishes.
        if f <= 165:
            draw_ball(img, bx + 40, 260)
        # Ball C: red robot's second ball (appears after it turns around).
        if 205 <= f <= 315:
            draw_ball(img, rx - 40, 260)
        # Ball D: rolls along the bottom and off the right edge -- never scores.
        if 60 <= f <= 140:
            dx = lerp(100, 980, (f - 60) / 80)
            draw_ball(img, dx, 500)

        writer.write(img)

    writer.release()

    zones = [
        {"name": "red_goal", "alliance": "red", "rect": list(RED_GOAL)},
        {"name": "blue_goal", "alliance": "blue", "rect": list(BLUE_GOAL)},
    ]
    with open("test_zones.json", "w") as fp:
        json.dump(zones, fp, indent=2)

    print(f"Wrote test_clip.mp4 ({FRAMES} frames @ {FPS}fps) and test_zones.json")
    print("Expected: Red R0 = 2 balls, Blue B0 = 1 ball, total 3 scored.")


if __name__ == "__main__":
    main()
