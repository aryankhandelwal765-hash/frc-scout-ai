"""Statbotics data loader — wraps the official `statbotics` Python library (v3).

The library is the documented, supported interface (pip install statbotics==3.0.0);
it targets the same backend as the REST API (https://api.statbotics.io/v3) but
handles query construction and field selection for us. parse_team_stats maps a
team_year record onto the flat schema the engines consume.

is_reachable() lets the UI tell "Statbotics is down/unreachable" apart from
"this team genuinely has no data for the year" — otherwise every failure looks
like a missing team and silently collapses to zero-EPA defaults.
"""

import warnings

import requests

_ROOT = "https://api.statbotics.io/v3/"
_client = None


def _sb():
    """Lazily import the `statbotics` package and construct a single client.

    Imported here rather than at module top so that merely importing this module
    (e.g. via the data_source dispatcher) never fails when the optional
    `statbotics` package isn't installed — only actually using it does.
    """
    global _client
    if _client is None:
        import statbotics  # optional dependency; only needed for this provider

        _client = statbotics.Statbotics()
    return _client


def is_reachable(timeout: float = 5.0) -> bool:
    """True if the Statbotics v3 API root responds (i.e. service is up)."""
    try:
        r = requests.get(_ROOT, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def fetch_team_year(team: int, year: int = 2026) -> dict | None:
    """Return the raw team_year record, or None on any failure.

    The library raises (UserWarning subclasses Exception) on an invalid/failed
    query and on network errors — all of which we treat as "no data".
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _sb().get_team_year(team, year)
    except Exception:
        return None


def parse_team_stats(data: dict) -> dict:
    epa = data.get("epa", {})
    total = epa.get("total_points", {})
    breakdown = epa.get("breakdown", {})
    stats = epa.get("stats", {})
    ranks = epa.get("ranks", {}).get("total", {})
    record = data.get("record", {})

    return {
        "team": data.get("team"),
        "name": data.get("name", "Unknown"),
        "epa": total.get("mean", 0.0),
        "epa_sd": total.get("sd", 0.0),
        "epa_max": stats.get("max", total.get("mean", 0.0)),
        "auto": breakdown.get("auto_points", 0.0),
        "teleop": breakdown.get("teleop_points", 0.0),
        "endgame": breakdown.get("endgame_points", 0.0),
        "winrate": record.get("winrate", 0.5),
        "wins": record.get("wins", 0),
        "losses": record.get("losses", 0),
        "ties": record.get("ties", 0),
        "count": record.get("count", 0),
        "rank": ranks.get("rank"),
        "percentile": ranks.get("percentile"),
        "team_count": ranks.get("team_count"),
    }


def get_team_stats(team: int, year: int = 2026) -> dict | None:
    data = fetch_team_year(team, year)
    if data is None:
        return None
    return parse_team_stats(data)
