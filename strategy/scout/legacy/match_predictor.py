import os

# Directory holding the raw scouting CSV exports. Override with the
# SCOUTING_DATA_DIR env var; defaults to ~/Downloads.
SCOUTING_DATA_DIR = os.environ.get(
    "SCOUTING_DATA_DIR", os.path.expanduser("~/Downloads")
)

import pandas as pd
import joblib

from scout.legacy.alliance_synergy import analyze_alliance

# =========================================================
# CONFIG
# =========================================================

CSV_PATH = os.path.join(SCOUTING_DATA_DIR, "scouting_with_metrics_20260508_123118.csv")
MODEL_PATH = (
    "models/score_predictor.pkl"
)

SCALER_PATH = (
    "models/feature_scaler.pkl"
)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CSV_PATH)

model = joblib.load(
    MODEL_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

df["Team Number"] = pd.to_numeric(
    df["Team Number"],
    errors="coerce"
)

# =========================================================
# HELPERS
# =========================================================

def get_robot(team_number):

    matching = df[
        df["Team Number"] == team_number
    ]

    if matching.empty:

        raise Exception(
            f"Team {team_number} not found."
        )

    return matching.iloc[0]


def build_alliance(team_list):

    robots = []

    for team in team_list:

        robots.append(
            get_robot(team)
        )

    return robots


# =========================================================
# DEFENSE PRESSURE MODEL
# =========================================================




# =========================================================
# MATCHUP ADJUSTMENT
# =========================================================



# =========================================================
# WIN PROBABILITY
# =========================================================

def calculate_win_probability(
    red_score,
    blue_score
):

    margin = red_score - blue_score

    probability = 50 + (margin * 1.1)

    probability = max(
        1,
        min(99, probability)
    )

    return round(probability, 1)


# =========================================================
# MATCH PREDICTION
# =========================================================
def predict_alliance_score(
    alliance_analysis
):
    features = pd.DataFrame([{
        "offensive_power":
            alliance_analysis[
                "offensive_power"
            ],

        "defensive_stability":
            alliance_analysis[
                "defensive_stability"
            ],

        "role_balance":
            alliance_analysis[
                "role_balance"
            ],

        "route_diversity":
            alliance_analysis[
                "route_diversity"
            ],

        "traffic_risk":
            alliance_analysis[
                "traffic_risk"
            ],

        "overall_synergy":
            alliance_analysis[
                "overall_synergy"
            ]
    }])


    scaled_features = scaler.transform(
        features
    )

    prediction = model.predict(
        scaled_features
    )[0]

    return round(
        max(0, prediction),
        1
    )

def predict_match(
    red_teams,
    blue_teams
):

    red_alliance = build_alliance(
        red_teams
    )

    blue_alliance = build_alliance(
        blue_teams
    )

    # =====================================================
    # ANALYZE BOTH ALLIANCES
    # =====================================================

    red_analysis = analyze_alliance(
        red_alliance
    )

    blue_analysis = analyze_alliance(
        blue_alliance
    )

    # =====================================================
    # APPLY MATCHUP EFFECTS
    # =====================================================

    # =====================================================
    # ML SCORE PREDICTION
    # =====================================================

    red_final = predict_alliance_score(
        red_analysis
    )

    blue_final = predict_alliance_score(
        blue_analysis
    )

    # =====================================================
    # DETERMINE WINNER
    # =====================================================

    if red_final > blue_final:

        winner = "RED"

    else:

        winner = "BLUE"

    margin = round(
        abs(red_final - blue_final),
        1
    )

    red_win_probability = (
        calculate_win_probability(
            red_final,
            blue_final
        )
    )

    blue_win_probability = round(
        100 - red_win_probability,
        1
    )

    return {

        "red_score":
            red_final,

        "blue_score":
            blue_final,

        "winner":
            winner,

        "margin":
            margin,

        "red_win_probability":
            red_win_probability,

        "blue_win_probability":
            blue_win_probability,

        "red_analysis":
            red_analysis,

        "blue_analysis":
            blue_analysis
    }


# =========================================================
# TEST MATCH
# =========================================================

if __name__ == "__main__":

    RED = [
        2056,
        581,
        1787
    ]

    BLUE = [
        6329,
        503,
        401
    ]

    result = predict_match(
        RED,
        BLUE
    )

    print("\n" + "=" * 80)

    print("MATCH PREDICTION\n")

    print(f"RED:  {RED}")
    print(f"BLUE: {BLUE}")

    print("\n" + "=" * 80)

    print(f"\nRED SCORE:  {result['red_score']}")
    print(f"BLUE SCORE: {result['blue_score']}")

    print(f"\nWINNER: {result['winner']}")

    print(f"MARGIN: {result['margin']}")

    print(
        f"\nRED WIN %: "
        f"{result['red_win_probability']}"
    )

    print(
        f"BLUE WIN %: "
        f"{result['blue_win_probability']}"
    )

    print("\n" + "=" * 80 + "\n")