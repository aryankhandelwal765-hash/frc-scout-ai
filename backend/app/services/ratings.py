"""Ten-season team ratings (built by legacy/strategy/scout/training/history_elo.py).

Reads the generated JSON file; every function returns None when it is absent so the app works
without it. Unknown teams count as an average team (BASE) and are reported to the caller.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = 1500.0
_cache: dict = {}


def load(path: str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    key = (str(p), p.stat().st_mtime)
    if _cache.get("key") != key:
        data = json.loads(p.read_text())
        ordered = sorted(data["ratings"].items(), key=lambda kv: -kv[1])
        data["rank"] = {t: i + 1 for i, (t, _r) in enumerate(ordered)}
        _cache.update(key=key, data=data)
    return _cache["data"]


def rating_and_rank(path: str, team: int) -> tuple[float | None, int | None]:
    d = load(path)
    if d is None or str(team) not in d["ratings"]:
        return None, None
    return float(d["ratings"][str(team)]), d["rank"][str(team)]


def win_prob(path: str, red: list[int], blue: list[int]) -> dict | None:
    d = load(path)
    if d is None:
        return None
    r = [float(d["ratings"].get(str(t), BASE)) for t in red]
    b = [float(d["ratings"].get(str(t), BASE)) for t in blue]
    diff = sum(r) / len(r) - sum(b) / len(b)
    p = 1.0 / (1.0 + 10 ** (-diff / d["logistic_scale"]))
    return {
        "red_win_prob": p,
        "blue_win_prob": 1.0 - p,
        "rating_diff": diff,
        "unknown_teams": [t for t in [*red, *blue] if str(t) not in d["ratings"]],
    }
