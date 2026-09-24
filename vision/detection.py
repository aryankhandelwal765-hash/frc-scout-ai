"""
Detection: turn a frame into robot blobs and ball blobs using classic CV.

Robots  -> alliance-colored bumper blobs (large, any shape).
Balls   -> game-piece-colored blobs that are ROUND (circularity filter).

No training, CPU-only. This is the deliberately simple layer you'd later swap
for a fine-tuned YOLO model when you outgrow color thresholding.
"""

import cv2
import numpy as np


def get_color_mask(hsv, ranges, kernel_size=5):
    """Binary mask of pixels whose HSV falls in any of `ranges`, opened to
    knock out salt-and-pepper noise from compression / crowd colors."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def get_ball_mask(hsv, ranges, close_kernel=2):
    """Color mask for BALLS. Unlike get_color_mask this does NOT do a
    morphological OPEN -- opening erodes small distant balls (~8-15px) right off
    the mask. Instead an optional tiny CLOSE fills speckle inside each ball."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    if close_kernel > 0:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def find_robot_blobs(mask, min_area):
    """Contours above `min_area`. Robots can be any shape, so no circularity gate."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        blobs.append({
            "bbox": (x, y, w, h),
            "centroid": (x + w / 2.0, y + h / 2.0),
            "area": area,
        })
    return blobs


def _circularity(contour, area):
    """4*pi*area / perimeter^2. 1.0 for a perfect circle, lower for ragged blobs."""
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4.0 * np.pi * area / (perimeter * perimeter)


def find_ball_blobs(mask, min_area, max_area, min_circularity):
    """Contours in the ball's color range that are round enough to be a ball.

    The circularity gate is what separates a round game piece from a same-colored
    but non-round object (a bumper of the same color, a banner, a jersey)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        if _circularity(c, area) < min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        x, y, w, h = cv2.boundingRect(c)
        blobs.append({
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
            "radius": float(radius),
            "area": area,
        })
    return blobs
