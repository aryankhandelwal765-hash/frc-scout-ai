"""Pytest bootstrap.

Lives at the repo root so the source trees are importable without an
install step or a PYTHONPATH export:

  strategy/               -> `scout.*`
  vision/match_pipeline/  -> `stitch`, `topdown`, `scoring_estimator`, ...

The vision tree is a flat script directory by design (no __init__.py), so it goes
on sys.path directly, which is exactly what the scripts themselves do.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRATEGY = ROOT / "strategy"
VISION_PIPELINE = ROOT / "vision" / "match_pipeline"

# ROOT so `from tests.conftest import ...` resolves regardless of invocation dir.
for p in (ROOT, STRATEGY, VISION_PIPELINE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
