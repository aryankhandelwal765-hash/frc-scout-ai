import math


def _alliance_stats(team_stats_list: list[dict]) -> dict:
    """Average and max EPA stats across an alliance of 3 teams."""
    n = len(team_stats_list)
    if n == 0:
        return {}

    def avg(key):
        return sum(t.get(key, 0) for t in team_stats_list) / n

    def mx(key):
        return max(t.get(key, 0) for t in team_stats_list)

    return {
        "epa": avg("epa"),
        "epa_max": mx("epa"),
        "epa_sd": avg("epa_sd"),
        "auto": avg("auto"),
        "teleop": avg("teleop"),
        "endgame": avg("endgame"),
        "winrate": avg("winrate"),
        "sd": avg("epa_sd"),
    }


def build_matchup_features(
    red_stats: list[dict],
    blue_stats: list[dict],
) -> dict:
    """Build the full MODEL_FEATURES vector from team stat dicts.

    Scouting-only features default to neutral values (0 for diffs).
    """
    r = _alliance_stats(red_stats)
    b = _alliance_stats(blue_stats)

    def diff(key):
        return r.get(key, 0) - b.get(key, 0)

    features = {
        "red_epa": r["epa"],
        "blue_epa": b["epa"],
        "epa_diff": diff("epa"),
        "red_epa_max": r["epa_max"],
        "blue_epa_max": b["epa_max"],
        "epa_max_diff": diff("epa_max"),
        "red_auto": r["auto"],
        "blue_auto": b["auto"],
        "auto_diff": diff("auto"),
        "red_teleop": r["teleop"],
        "blue_teleop": b["teleop"],
        "teleop_diff": diff("teleop"),
        "red_endgame": r["endgame"],
        "blue_endgame": b["endgame"],
        "endgame_diff": diff("endgame"),
        "red_winrate": r["winrate"],
        "blue_winrate": b["winrate"],
        "winrate_diff": diff("winrate"),
        "red_sd": r["sd"],
        "blue_sd": b["sd"],
        "sd_diff": diff("sd"),
        # Scouting features — default to neutral (0 diff = no edge)
        "red_avg_driver_skill": 5.0,
        "blue_avg_driver_skill": 5.0,
        "avg_driver_skill_diff": 0.0,
        "red_avg_reliability": 5.0,
        "blue_avg_reliability": 5.0,
        "avg_reliability_diff": 0.0,
        "red_defense_frequency": 0.0,
        "blue_defense_frequency": 0.0,
        "defense_frequency_diff": 0.0,
        "red_climb_success_rate": 0.5,
        "blue_climb_success_rate": 0.5,
        "climb_success_rate_diff": 0.0,
        "red_failed_climb_rate": 0.1,
        "blue_failed_climb_rate": 0.1,
        "failed_climb_rate_diff": 0.0,
        "red_trench_usage_rate": 0.33,
        "blue_trench_usage_rate": 0.33,
        "trench_usage_rate_diff": 0.0,
        "red_bump_usage_rate": 0.33,
        "blue_bump_usage_rate": 0.33,
        "bump_usage_rate_diff": 0.0,
        "red_passing_usage_rate": 0.33,
        "blue_passing_usage_rate": 0.33,
        "passing_usage_rate_diff": 0.0,
        "red_avg_hp_accuracy": 0.5,
        "blue_avg_hp_accuracy": 0.5,
        "avg_hp_accuracy_diff": 0.0,
        "red_avg_auto_fuel": 0.0,
        "blue_avg_auto_fuel": 0.0,
        "avg_auto_fuel_diff": 0.0,
        "red_scouting_confidence": 0.0,
        "blue_scouting_confidence": 0.0,
        "scouting_confidence_diff": 0.0,
        "trench_congestion_diff": 0.0,
        "bump_congestion_diff": 0.0,
        "passing_synergy_diff": 0.0,
        "defensive_flexibility_diff": 0.0,
        "scorer_overlap_diff": 0.0,
    }
    return features


def win_probability_from_diff(score_diff: float, sigma: float = 35.0) -> float:
    """Convert predicted score differential to red win probability (0–1)."""
    return 1.0 / (1.0 + math.exp(-score_diff / sigma))
