"""Pytest bootstrap.

Lives at the repo root so the three source trees are importable without an
install step or a PYTHONPATH export:

  strategy/               -> `scout.*`
  vision/classic/         -> `track_match`, `scoring`, `detection`, ...
  vision/match_pipeline/  -> `stitch`, `topdown`, `compare_tba`, ...

The two vision trees are flat script directories by design (no __init__.py), so
they go on sys.path directly, which is exactly what the scripts themselves do.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRATEGY = ROOT / "strategy"
VISION_CLASSIC = ROOT / "vision" / "classic"
VISION_PIPELINE = ROOT / "vision" / "match_pipeline"

# ROOT so `from tests.conftest import ...` resolves regardless of invocation dir.
for p in (ROOT, STRATEGY, VISION_CLASSIC, VISION_PIPELINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
