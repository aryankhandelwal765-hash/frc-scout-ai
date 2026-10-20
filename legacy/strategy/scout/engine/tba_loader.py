"""The Blue Alliance (TBA) data loader.

Statbotics' data API has been down since 2026-07-09 (their issues #414/#418), so
this provides an alternate source. TBA has no EPA, so we build the same flat
stats schema the engines consume out of TBA primitives:

    epa        <- mean OPR across the team's events (OPR = offensive contribution)
    epa_max    <- best single-event OPR
    epa_sd     <- spread of per-event OPRs
    auto/teleop/endgame <- OPR split by the team's alliance phase shares
    wins/losses/ties/winrate <- from the team's season matches
    rank/percentile/team_count <- None (no global EPA ranking in TBA)

Public interface mirrors statbotics_loader so it is a drop-in provider:
    is_configured(), is_reachable(), get_team_stats(team, year).

Requires a free TBA read key (config.get_tba_key), from the TBA_AUTH_KEY
environment variable.
"""

import statistics
import time

import requests

from scout.config import TBA_BASE, get_tba_key

# Per-process cache: many teams share events, so cache OPR payloads by event key.
_oprs_cache: dict[str, dict] = {}

# Short TTL cache for assembled team stats — the frontend calls several endpoints
# per analysis (predict, read, coach, bots), each of which would otherwise refetch
# every team from TBA. 10 min keeps things fresh across an event while avoiding
# redundant calls / rate limits.
_stats_cache: dict[tuple[int, int], tuple[float, dict | None]] = {}
_STATS_TTL = 600.0


def is_configured() -> bool:
    return get_tba_key() is not None


def _headers() -> dict:
    return {"X-TBA-Auth-Key": get_tba_key() or "", "accept": "application/json"}


def is_reachable(timeout: float = 5.0) -> bool:
    """True if TBA responds and the key is accepted (probes /status)."""
    if not is_configured():
        return False
    try:
        r = requests.get(f"{TBA_BASE}/status", headers=_headers(), timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _get(path: str, timeout: float = 10.0):
    """GET a TBA path, return parsed JSON or None on any failure."""
    try:
        r = requests.get(f"{TBA_BASE}{path}", headers=_headers(), timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def _team_name(team: int) -> str:
    data = _get(f"/team/frc{team}")
    if isinstance(data, dict):
        return data.get("nickname") or data.get("name") or f"Team {team}"
    return f"Team {team}"


def _event_oprs(event_key: str) -> dict:
    """OPR payload for an event: {'oprs': {...}, 'dprs': {...}, 'ccwms': {...}}."""
    if event_key not in _oprs_cache:
        _oprs_cache[event_key] = _get(f"/event/{event_key}/oprs") or {}
    return _oprs_cache[event_key]


def _phase_points(breakdown: dict) -> tuple[float, float, float]:
    """Best-effort (auto, teleop, endgame) points from an alliance score breakdown.

    Game-agnostic: TBA breakdown keys change every year. Recent games (incl. 2026)
    use total{Auto,Teleop}Points; older ones use {auto,teleop}Points. The endgame
    aggregate key varies, so we try known names then fall back to any key matching
    'endgame'+'points'. teleopPoints includes endgame, so we subtract it back out.
    Only phase *shares* are used downstream, so small misattributions are tolerable
    and a missing breakdown just yields zeros.

    Verified against real 2026 data: totalAutoPoints + totalTeleopPoints =
    totalPoints - foulPoints; endgame lives in endGameTowerPoints (part of teleop).
    """
    if not isinstance(breakdown, dict):
        return 0.0, 0.0, 0.0

    low = {k.lower(): v for k, v in breakdown.items()}

    def pick(*names):
        for n in names:
            v = low.get(n.lower())
            if isinstance(v, (int, float)):
                return float(v)
        return None

    auto = pick("autoPoints", "totalAutoPoints") or 0.0
    teleop = pick("teleopPoints", "totalTeleopPoints") or 0.0

    # 2026 endgame/climb is worth ~nothing this season — hard-zero it so no
    # downstream engine (archetype, chemistry, LLM prompts) weights it. Do NOT
    # subtract it out of teleop either — leave those cycles counting as teleop.
    endgame = 0.0

    return max(auto, 0.0), max(teleop, 0.0), endgame


def _team_color(match: dict, team: int) -> str | None:
    key = f"frc{team}"
    for color in ("red", "blue"):
        if key in match.get("alliances", {}).get(color, {}).get("team_keys", []):
            return color
    return None


def fetch_team_stats(team: int, year: int = 2026) -> dict | None:
    """Aggregate a team's TBA season into the engine stats schema, or None."""
    if not is_configured():
        return None

    event_keys = _get(f"/team/frc{team}/events/{year}/keys")
    matches = _get(f"/team/frc{team}/matches/{year}")

    # No event participation at all -> treat as no data (distinct from a bad key,
    # which is_reachable() catches at the UI layer).
    if not event_keys and not matches:
        return None

    # --- OPR across events -> EPA proxy ----------------------------------------
    oprs = []
    for ek in event_keys or []:
        val = _event_oprs(ek).get("oprs", {}).get(f"frc{team}")
        if isinstance(val, (int, float)):
            oprs.append(float(val))

    epa = statistics.mean(oprs) if oprs else 0.0
    epa_max = max(oprs) if oprs else epa
    epa_sd = statistics.stdev(oprs) if len(oprs) >= 2 else 0.0

    # --- Phase shares + record from matches ------------------------------------
    auto_sum = teleop_sum = endgame_sum = 0.0
    phase_matches = 0
    wins = losses = ties = 0

    for m in matches or []:
        color = _team_color(m, team)
        if color is None:
            continue
        score = m.get("alliances", {}).get(color, {}).get("score", -1)
        if score is None or score < 0:
            continue  # not yet played
        # record
        winner = m.get("winning_alliance", "")
        if winner == "":
            ties += 1
        elif winner == color:
            wins += 1
        else:
            losses += 1
        # phase shares from this alliance's breakdown
        bd = (m.get("score_breakdown") or {}).get(color)
        a, t, e = _phase_points(bd)
        if (a + t + e) > 0:
            total = a + t + e
            auto_sum += a / total
            teleop_sum += t / total
            endgame_sum += e / total
            phase_matches += 1

    if phase_matches > 0:
        auto = epa * (auto_sum / phase_matches)
        teleop = epa * (teleop_sum / phase_matches)
        endgame = epa * (endgame_sum / phase_matches)
    else:
        auto = teleop = endgame = 0.0

    count = wins + losses + ties
    winrate = (wins / count) if count > 0 else 0.5

    return {
        "team": team,
        "name": _team_name(team),
        "epa": round(epa, 2),
        "epa_sd": round(epa_sd, 2),
        "epa_max": round(epa_max, 2),
        "auto": round(auto, 2),
        "teleop": round(teleop, 2),
        "endgame": round(endgame, 2),
        "winrate": round(winrate, 3),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "count": count,
        "rank": None,
        "percentile": None,
        "team_count": None,
    }


def get_team_stats(team: int, year: int = 2026) -> dict | None:
    """Drop-in replacement for statbotics_loader.get_team_stats.

    Two-tier cache: in-memory dict (fast, 10 min) → SQLite (persistent, 24 h) → TBA.
    The persistent layer survives API restarts and Docker rebuilds.
    """
    key = (team, year)
    hit = _stats_cache.get(key)
    if hit and (time.time() - hit[0]) < _STATS_TTL:
        return hit[1]

    from scout.db import TTL_TBA, cache_get, cache_set, tba_team_key

    db_key = tba_team_key(team, year)
    db_hit = cache_get(db_key)
    if db_hit is not None:
        _stats_cache[key] = (time.time(), db_hit)
        return db_hit

    stats = fetch_team_stats(team, year)
    _stats_cache[key] = (time.time(), stats)
    if stats is not None:
        cache_set(db_key, stats, TTL_TBA)
    return stats


# ── Match import ────────────────────────────────────────────────────────────────

# Ordering for comp levels so quals sort before playoffs.
_LEVEL_ORDER = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}


def _match_teams(match: dict, color: str) -> list[int]:
    keys = match.get("alliances", {}).get(color, {}).get("team_keys", [])
    out = []
    for k in keys:
        try:
            out.append(int(k.replace("frc", "")))
        except ValueError:
            pass
    return out


def _match_label(match: dict) -> str:
    cl = match.get("comp_level", "?")
    names = {"qm": "Qual", "ef": "Eighth", "qf": "Quarter", "sf": "Semi", "f": "Final"}
    base = names.get(cl, cl.upper())
    num = match.get("match_number", "?")
    if cl in ("sf", "qf", "ef") and match.get("set_number"):
        return f"{base} {match['set_number']}-{num}"
    return f"{base} {num}"


def get_event_matches(event_key: str) -> list[dict]:
    """Return this event's matches as [{key,label,comp_level,match_number,red,blue}].

    Sorted quals-first then by match number. Empty list on any failure (bad key,
    no key configured, unreachable, or event not found). SQLite-cached for 24h.
    """
    if not is_configured():
        return []

    from scout.db import TTL_TBA, cache_get, cache_set, tba_matches_key

    db_key = tba_matches_key(event_key)
    db_hit = cache_get(db_key)
    if db_hit is not None:
        return db_hit

    data = _get(f"/event/{event_key}/matches/simple")
    if not isinstance(data, list):
        return []
    matches = []
    for m in data:
        matches.append(
            {
                "key": m.get("key"),
                "label": _match_label(m),
                "comp_level": m.get("comp_level"),
                "match_number": m.get("match_number", 0),
                "set_number": m.get("set_number", 0),
                "red": _match_teams(m, "red"),
                "blue": _match_teams(m, "blue"),
            }
        )
    matches.sort(
        key=lambda x: (
            _LEVEL_ORDER.get(x["comp_level"], 9),
            x["set_number"] or 0,
            x["match_number"] or 0,
        )
    )
    # Only matches with a full 3v3 lineup are usable for prediction.
    result = [m for m in matches if len(m["red"]) == 3 and len(m["blue"]) == 3]
    cache_set(db_key, result, TTL_TBA)
    return result
