"""
Pure-python post-filters for robot detections (no cv2/torch, so they unit-test cheaply).

  veto_people          drop robot boxes that sit on a person (ref, FTA, driver)
  merge_nested         collapse a small box fully inside a bigger one (duplicate)
  eliminate_alliances  fill unreadable bumper colours from the 3-per-alliance rule

Boxes are (x1, y1, x2, y2) in whatever pixel space the caller uses.
"""

ALLIANCE_SIZE = 3


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def contained_frac(inner, outer):
    """Share of `inner`'s area that lies inside `outer` (0..1)."""
    a = _area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def veto_people(boxes, people, thresh=0.6):
    """Split robot boxes into (kept, vetoed).

    A box is vetoed when at least `thresh` of ITS OWN area is inside a person box.
    Containment of the robot box, not IoU: a robot half hidden behind a standing ref
    overlaps the ref's tall box by well under `thresh` and survives, while a box drawn
    around the ref itself is ~100% inside it and goes.
    """
    kept, vetoed = [], []
    for b in boxes:
        (vetoed if any(contained_frac(b, p) >= thresh for p in people) else kept).append(b)
    return kept, vetoed


def merge_nested(boxes, thresh=0.85):
    """Drop a box that is >= `thresh` inside a larger one (keeps the larger).

    IoU-based NMS misses these: a small box nested in a big one has low IoU with it.
    `boxes` are dicts with a "box" key; order of survivors is preserved.
    """
    keep = []
    for i, d in enumerate(boxes):
        nested = any(
            j != i
            and _area(o["box"]) > _area(d["box"])
            and contained_frac(d["box"], o["box"]) >= thresh
            for j, o in enumerate(boxes)
        )
        if not nested:
            keep.append(d)
    return keep


def eliminate_alliances(colours):
    """Fill unknown colours using the 3-robots-per-alliance rule.

    `colours` is a list of "blue" | "red" | None, one per robot in a frame.
    Returns (filled, how) with how[i] in {"read", "elim", "unknown"}.

    - three robots already read as one colour: every unknown must be the other colour
    - otherwise the count cannot decide, so the unknown stays None (do not guess)
    - if the other colour is already full too, the frame is impossible (detector
      error): unknowns stay None
    """
    nb = sum(c == "blue" for c in colours)
    nr = sum(c == "red" for c in colours)
    filled, how = [], []
    for c in colours:
        if c is not None:
            filled.append(c)
            how.append("read")
        elif nb >= ALLIANCE_SIZE and nr < ALLIANCE_SIZE:
            filled.append("red")
            how.append("elim")
        elif nr >= ALLIANCE_SIZE and nb < ALLIANCE_SIZE:
            filled.append("blue")
            how.append("elim")
        else:
            filled.append(None)
            how.append("unknown")
    return filled, how
