"""Shared frame geometry for the vision pipeline.

These boxes used to be copy-pasted into six modules, so re-tuning one feed meant
finding and editing every copy. They live here now and every module imports them.

Both values are overridable from the environment, which is what you want when
pointing the pipeline at a broadcast with different framing:

    VISION_CROP="0,100,1280,470"
    VISION_TIMER_BOX_FRAC="0.4688,0.0625,0.5352,0.1250"

NOTE: CROP is a *fallback* for 1280x720 footage. The live path is
``trajectories.field_crop(frame.shape)``, which is resolution-independent and
should be preferred. CROP is only used when the frame cannot be probed.
"""

import os


def _tuple_from_env(name: str, default: tuple) -> tuple:
    """Parse "a,b,c,d" from the environment, falling back to `default`.

    A malformed value falls back rather than raising: a bad env var should not
    take down a batch run that is otherwise fine.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        parts = tuple(type(d)(p.strip()) for d, p in zip(default, raw.split(",")))
    except (TypeError, ValueError):
        return default
    return parts if len(parts) == len(default) else default


# Detector crop: drops the stitched second camera angle and the sponsor banner.
# MUST match what the YOLO detector was trained on.
CROP = _tuple_from_env("VISION_CROP", (0, 100, 1280, 470))

# Centre match timer "M:SS", as FRACTIONS of frame width/height so it holds at
# both 720p and 1080p.
TIMER_BOX_FRAC = _tuple_from_env("VISION_TIMER_BOX_FRAC", (0.4688, 0.0625, 0.5352, 0.1250))
