"""Volatility engine — consistency and risk profile from Statbotics EPA stats.

Adapts the original playoff-stability / reliability concepts (removed in the public release)
onto EPA: how much a team's contribution swings match-to-match, and how much
upside (ceiling) it carries. Consumes the stats dict from statbotics_loader.

Key inputs: epa (mean contribution), epa_sd (match-to-match spread),
epa_max (best observed single-match contribution).
"""


def _clamp(value: float, lo: float = 1.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, round(value, 2)))


def profile_volatility(stats: dict) -> dict:
    """Return a consistency / risk profile for one team.

    - coefficient of variation (cv) = epa_sd / epa — normalized swing.
    - floor / ceiling = one standard deviation below the mean and the best
      observed match, i.e. realistic bad-day and good-day contributions.
    - consistency_score (1-10): higher = more reliable, lower = streakier.
    """
    epa = max(stats.get("epa", 0.0), 0.0)
    sd = max(stats.get("epa_sd", 0.0), 0.0)
    epa_max = max(stats.get("epa_max", epa), epa)

    cv = (sd / epa) if epa > 0 else 0.0
    floor = max(epa - sd, 0.0)
    ceiling = max(epa_max, epa + sd)
    ceiling_gap = ceiling - epa

    # Consistency: cv of 0 -> 10, cv of ~0.6+ -> ~1. Linear map, then clamp.
    consistency_score = _clamp(10.0 - (cv / 0.6) * 9.0)

    if cv <= 0.15:
        rating = "Metronome"          # extremely consistent
    elif cv <= 0.30:
        rating = "Steady"
    elif cv <= 0.45:
        rating = "Streaky"
    else:
        rating = "Boom-or-Bust"

    return {
        "team": stats.get("team"),
        "name": stats.get("name", "Unknown"),
        "epa": round(epa, 1),
        "epa_sd": round(sd, 1),
        "cv": round(cv, 3),
        "floor": round(floor, 1),
        "ceiling": round(ceiling, 1),
        "ceiling_gap": round(ceiling_gap, 1),
        "consistency_score": consistency_score,
        "volatility_rating": rating,
        "summary": (
            f"Team {stats.get('team')}: {rating} "
            f"(consistency {consistency_score:.1f}/10) — "
            f"floor ~{floor:.0f}, expected {epa:.0f}, ceiling ~{ceiling:.0f}."
        ),
    }


def profile_alliance(stats_list: list[dict]) -> dict:
    """Aggregate volatility across an alliance of 3 teams.

    Alliance variance is the sum of per-team variances (treating team swings as
    independent), so alliance sd = sqrt(sum(sd^2)). A high-floor / high-ceiling
    read tells the strategy layer whether an alliance is safe or swingy.
    """
    teams = [profile_volatility(s) for s in stats_list]
    if not teams:
        return {"teams": [], "alliance_floor": 0.0, "alliance_ceiling": 0.0,
                "alliance_sd": 0.0, "alliance_consistency": 0.0}

    alliance_floor = sum(t["floor"] for t in teams)
    alliance_ceiling = sum(t["ceiling"] for t in teams)
    alliance_sd = sum(t["epa_sd"] ** 2 for t in teams) ** 0.5
    alliance_consistency = round(sum(t["consistency_score"] for t in teams) / len(teams), 2)

    return {
        "teams": teams,
        "alliance_floor": round(alliance_floor, 1),
        "alliance_ceiling": round(alliance_ceiling, 1),
        "alliance_sd": round(alliance_sd, 1),
        "alliance_consistency": alliance_consistency,
    }
