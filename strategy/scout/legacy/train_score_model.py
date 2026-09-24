import os

# Directory holding the raw scouting CSV exports. Override with the
# SCOUTING_DATA_DIR env var; defaults to ~/Downloads.
SCOUTING_DATA_DIR = os.environ.get(
    "SCOUTING_DATA_DIR", os.path.expanduser("~/Downloads")
)

import json
import joblib
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scout.legacy.alliance_synergy import analyze_alliance

# =========================================================
# CONFIG
# =========================================================

MATCH_FILE = (
    "data/raw_matches/"
    "2026_worlds_all_matches.json"
)

SCOUTING_CSV = os.path.join(SCOUTING_DATA_DIR, "scouting_with_metrics_20260508_123118.csv")

MODEL_OUTPUT = (
    "models/score_predictor.pkl"
)

SCALER_OUTPUT = (
    "models/feature_scaler.pkl"
)

# =========================================================
# LOAD SCOUTING DATA
# =========================================================

print("\nLoading scouting database...\n")

df = pd.read_csv(SCOUTING_CSV)

df["Team Number"] = pd.to_numeric(
    df["Team Number"],
    errors="coerce"
)

# =========================================================
# LOAD MATCHES
# =========================================================

print("Loading match history...\n")

with open(MATCH_FILE, "r") as f:

    matches = json.load(f)

print(
    f"Loaded {len(matches)} matches.\n"
)

# =========================================================
# HELPERS
# =========================================================

def get_robot(team_number):

    matching = df[
        df["Team Number"] == team_number
    ]

    if matching.empty:

        return None

    return matching.iloc[0]


def build_alliance(team_list):

    robots = []

    for team in team_list:

        robot = get_robot(team)

        if robot is None:

            return None

        robots.append(robot)

    return robots

# =========================================================
# BUILD TRAINING DATASET
# =========================================================

training_rows = []

print("Generating alliance training samples...\n")

for match in matches:

    # =====================================================
    # PLAYOFF WEIGHTING
    # =====================================================

    sample_weight = 1.0

    if match["comp_level"] in [
        "qf",
        "sf",
        "f"
    ]:
        sample_weight = 2.5

    # =====================================================
    # RED ALLIANCE
    # =====================================================

    red_robots = build_alliance(
        match["red_teams"]
    )

    if red_robots is not None:

        analysis = analyze_alliance(
            red_robots
        )

        training_rows.append({

            "offensive_power":
                analysis["offensive_power"],

            "defensive_stability":
                analysis[
                    "defensive_stability"
                ],

            "role_balance":
                analysis[
                    "role_balance"
                ],

            "route_diversity":
                analysis[
                    "route_diversity"
                ],

            "traffic_risk":
                analysis[
                    "traffic_risk"
                ],

            "overall_synergy":
                analysis[
                    "overall_synergy"
                ],

            "actual_score":
                match["red_score"],

            "sample_weight":
                sample_weight
        })

    # =====================================================
    # BLUE ALLIANCE
    # =====================================================

    blue_robots = build_alliance(
        match["blue_teams"]
    )

    if blue_robots is not None:

        analysis = analyze_alliance(
            blue_robots
        )

        training_rows.append({

            "offensive_power":
                analysis["offensive_power"],

            "defensive_stability":
                analysis[
                    "defensive_stability"
                ],

            "role_balance":
                analysis[
                    "role_balance"
                ],

            "route_diversity":
                analysis[
                    "route_diversity"
                ],

            "traffic_risk":
                analysis[
                    "traffic_risk"
                ],

            "overall_synergy":
                analysis[
                    "overall_synergy"
                ],

            "actual_score":
                match["blue_score"],

            "sample_weight":
                sample_weight
        })

# =========================================================
# DATAFRAME
# =========================================================

train_df = pd.DataFrame(
    training_rows
)

print(
    f"Generated "
    f"{len(train_df)} alliance samples.\n"
)

# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [

    "offensive_power",

    "defensive_stability",

    "role_balance",

    "route_diversity",

    "traffic_risk",

    "overall_synergy"
]

X = train_df[
    FEATURE_COLUMNS
]

y = train_df[
    "actual_score"
]

weights = train_df[
    "sample_weight"
]

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test, w_train, w_test = (
    train_test_split(
        X,
        y,
        weights,
        test_size=0.2,
        random_state=42
    )
)

# =========================================================
# NORMALIZATION
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =========================================================
# MODEL
# =========================================================

print("Training Ridge regression model...\n")

model = Ridge(alpha=10.0)

model.fit(
    X_train_scaled,
    y_train,
    sample_weight=w_train
)

# =========================================================
# EVALUATION
# =========================================================

predictions = model.predict(
    X_test_scaled
)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("=" * 80)

print("\nMODEL RESULTS\n")

print(
    f"Mean Absolute Error: "
    f"{mae:.2f}"
)

print("\nFeature Weights:\n")

for feature, coef in zip(
    FEATURE_COLUMNS,
    model.coef_
):

    print(
        f"{feature}: "
        f"{coef:.3f}"
    )

print(
    f"\nIntercept: "
    f"{model.intercept_:.3f}"
)

print("\n" + "=" * 80)

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    MODEL_OUTPUT
)

joblib.dump(
    scaler,
    SCALER_OUTPUT
)

print(
    f"\nSaved model:\n"
    f"{MODEL_OUTPUT}"
)

print(
    f"\nSaved scaler:\n"
    f"{SCALER_OUTPUT}\n"
)