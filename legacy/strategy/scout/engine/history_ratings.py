"""Win probability from ratings learned over ten seasons (scout/training/history_elo.py).

The ratings file is generated data (scout/data/history/team_ratings.json, gitignored). When it is
absent every function returns None and the callers fall back to what they did before.
"""

import json
import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "history" / "team_ratings.json"
BASE = 1500.0  # rating of a team the file has never seen (a typical team)
_cache: dict = {}


def _load() -> dict | None:
    path = Path(os.environ.get("HISTORY_RATINGS", DEFAULT_PATH))
    key = (str(path), path.stat().st_mtime) if path.is_file() else None
    if key is None:
        return None
    if _cache.get("key") != key:
        _cache.update(key=key, data=json.loads(path.read_text()))
    return _cache["data"]


def rating(team: int) -> float | None:
    d = _load()
    return None if d is None else float(d["ratings"].get(str(team), BASE))


def win_prob(red: list[int], blue: list[int]) -> dict | None:
    """Red's win probability, or None without a ratings file."""
    d = _load()
    if d is None or not red or not blue:
        return None
    r = [float(d["ratings"].get(str(t), BASE)) for t in red]
    b = [float(d["ratings"].get(str(t), BASE)) for t in blue]
    diff = sum(r) / len(r) - sum(b) / len(b)
    p = 1.0 / (1.0 + 10 ** (-diff / d["logistic_scale"]))
    unknown = [t for t in [*red, *blue] if str(t) not in d["ratings"]]
    return {
        "red_win_prob": p,
        "blue_win_prob": 1.0 - p,
        "rating_diff": diff,
        "unknown_teams": unknown,
    }
