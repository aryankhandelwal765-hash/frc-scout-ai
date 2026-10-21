"""Match prediction: EPA win probability plus the trained score model."""

from pathlib import Path

import joblib
import pandas as pd

from scout.engine import history_ratings
from scout.engine.data_source import get_team_stats
from scout.engine.feature_engineering import (
    build_matchup_features,
    win_probability_from_diff,
)
from scout.engine.model_features import MODEL_FEATURES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_score_model = None


def _load_model():
    global _score_model
    if _score_model is None:
        model_path = MODELS_DIR / "matchup_score_model.pkl"
        _score_model = joblib.load(model_path)
    return _score_model


def predict_match(
    red_teams: list[int],
    blue_teams: list[int],
    year: int = 2026,
) -> dict:
    all_teams = red_teams + blue_teams
    stats_map: dict[int, dict | None] = {}
    for team in all_teams:
        stats_map[team] = get_team_stats(team, year)

    missing = [t for t in all_teams if stats_map[t] is None]

    red_stats = [stats_map[t] or _default_stats(t) for t in red_teams]
    blue_stats = [stats_map[t] or _default_stats(t) for t in blue_teams]

    red_epa_sum = sum(s["epa"] for s in red_stats)
    blue_epa_sum = sum(s["epa"] for s in blue_stats)

    # EPA-based win probability (sigma=100 calibrated for 2026 score scale)
    epa_diff = red_epa_sum - blue_epa_sum
    red_win_prob = win_probability_from_diff(epa_diff, sigma=100.0)
    blue_win_prob = 1.0 - red_win_prob

    # ML model score differential (supplementary — can be noisy without scouting data)
    try:
        features = build_matchup_features(red_stats, blue_stats)
        df = pd.DataFrame([features])[
            MODEL_FEATURES
        ].values  # pass as array to avoid feature-name warning
        model_score_diff = float(_load_model().predict(df)[0])
    except Exception:
        model_score_diff = epa_diff

    history = history_ratings.win_prob(red_teams, blue_teams)

    return {
        "red_teams": red_teams,
        "blue_teams": blue_teams,
        "red_stats": red_stats,
        "blue_stats": blue_stats,
        "epa_diff": epa_diff,
        "model_score_diff": model_score_diff,
        "red_predicted_score": red_epa_sum,
        "blue_predicted_score": blue_epa_sum,
        "red_win_prob": red_win_prob,
        "blue_win_prob": blue_win_prob,
        "winner": "RED" if epa_diff > 0 else "BLUE",
        "missing_teams": missing,
        # Win probability from ten seasons of results; None when no ratings file is present.
        "history_win_prob": history,
    }


def _default_stats(team: int) -> dict:
    return {
        "team": team,
        "name": f"Team {team}",
        "epa": 0.0,
        "epa_sd": 0.0,
        "epa_max": 0.0,
        "auto": 0.0,
        "teleop": 0.0,
        "endgame": 0.0,
        "winrate": 0.5,
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "count": 0,
        "rank": None,
        "percentile": None,
        "team_count": None,
    }
