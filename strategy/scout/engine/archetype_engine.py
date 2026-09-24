"""Archetype engine — classify a team's playstyle from Statbotics EPA stats.

The legacy scouting-trait role logic (legacy/metric_engine.py) is the conceptual
blueprint, but scouting CSVs are empty, so archetypes are derived from the live
EPA breakdown (auto / teleop / endgame points) plus overall EPA and win rate.

Consumes the stats dict produced by statbotics_loader.parse_team_stats().
"""

# Phase-share thresholds (fraction of a team's scoring breakdown).
_AUTO_SPECIALIST = 0.28      # auto share this high = auto-focused
_ENDGAME_SPECIALIST = 0.30   # endgame share this high = climb/endgame-focused
_TELEOP_DOMINANT = 0.60      # teleop share this high = cycling workhorse
_BALANCED_SPREAD = 0.12      # max-min phase share below this = balanced


def _phase_shares(stats: dict) -> dict:
    """Return auto/teleop/endgame as fractions of total breakdown points."""
    auto = max(stats.get("auto", 0.0), 0.0)
    teleop = max(stats.get("teleop", 0.0), 0.0)
    endgame = max(stats.get("endgame", 0.0), 0.0)
    total = auto + teleop + endgame
    if total <= 0:
        return {"auto": 0.0, "teleop": 0.0, "endgame": 0.0}
    return {
        "auto": auto / total,
        "teleop": teleop / total,
        "endgame": endgame / total,
    }


def _tier(stats: dict) -> str:
    """Coarse strength tier from global percentile, falling back to raw EPA."""
    pct = stats.get("percentile")
    if pct is not None:
        if pct >= 0.90:
            return "Elite"
        if pct >= 0.70:
            return "Strong"
        if pct >= 0.40:
            return "Solid"
        return "Developing"
    # Fallback: rough 2026-scale EPA cutoffs when rank data is missing.
    epa = stats.get("epa", 0.0)
    if epa >= 75:
        return "Elite"
    if epa >= 50:
        return "Strong"
    if epa >= 30:
        return "Solid"
    return "Developing"


def classify_archetype(stats: dict) -> dict:
    """Classify one team's archetype from its EPA stats.

    Returns a dict with the archetype label, strength tier, dominant phase,
    the phase shares, and a one-line human summary suitable for LLM context.
    """
    shares = _phase_shares(stats)
    epa = stats.get("epa", 0.0)
    tier = _tier(stats)

    dominant = max(shares, key=shares.get) if any(shares.values()) else "none"
    spread = (max(shares.values()) - min(shares.values())) if any(shares.values()) else 0.0

    # Label precedence: specialists first, then balance, then teleop workhorse.
    if epa < 15 and tier == "Developing":
        label = "Support / Role Player"
    elif shares["endgame"] >= _ENDGAME_SPECIALIST:
        label = "Endgame Specialist"
    elif shares["auto"] >= _AUTO_SPECIALIST:
        label = "Auto Specialist"
    elif shares["teleop"] >= _TELEOP_DOMINANT:
        label = "Teleop Workhorse"
    elif spread <= _BALANCED_SPREAD:
        label = "Balanced Contributor"
    else:
        label = f"{dominant.capitalize()}-Leaning Contributor"

    return {
        "team": stats.get("team"),
        "name": stats.get("name", "Unknown"),
        "archetype": label,
        "tier": tier,
        "dominant_phase": dominant,
        "phase_shares": {k: round(v, 3) for k, v in shares.items()},
        "epa": round(epa, 1),
        "summary": (
            f"Team {stats.get('team')} ({stats.get('name', 'Unknown')}): "
            f"{tier} {label} — EPA {epa:.0f}, "
            f"auto {shares['auto']*100:.0f}% / teleop {shares['teleop']*100:.0f}% / "
            f"endgame {shares['endgame']*100:.0f}%."
        ),
    }


def classify_alliance(stats_list: list[dict]) -> list[dict]:
    """Classify every team on an alliance."""
    return [classify_archetype(s) for s in stats_list]
