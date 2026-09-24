import os

# Directory holding the raw scouting CSV exports. Override with the
# SCOUTING_DATA_DIR env var; defaults to ~/Downloads.
SCOUTING_DATA_DIR = os.environ.get(
    "SCOUTING_DATA_DIR", os.path.expanduser("~/Downloads")
)

import json

import pandas as pd
from match_predictor import predict_match

# =========================================================
# CONFIG
# =========================================================

MATCH_FILE = (
    "data/raw_matches/"
    "2026_worlds_all_matches.json"
)

SCOUTING_CSV = os.path.join(SCOUTING_DATA_DIR, "scouting_with_metrics_20260508_123118.csv")

# =========================================================
# LOAD SCOUTING DB
# =========================================================

df = pd.read_csv(SCOUTING_CSV)

df["Team Number"] = pd.to_numeric(
    df["Team Number"],
    errors="coerce"
)

available_teams = set(
    df["Team Number"]
    .dropna()
    .astype(int)
    .tolist()
)

# =========================================================
# LOAD MATCHES
# =========================================================

with open(MATCH_FILE, "r") as f:

    matches = json.load(f)

# =========================================================
# METRICS
# =========================================================

total_matches = 0

correct_winners = 0

total_score_error = 0

skipped_matches = 0

# =========================================================
# VALIDATION LOOP
# =========================================================

for match in matches:

    red = match["red_teams"]

    blue = match["blue_teams"]

    # =====================================================
    # Skip incomplete scouting coverage
    # =====================================================

    all_teams = red + blue

    missing = False

    for team in all_teams:

        if team not in available_teams:

            missing = True
            break

    if missing:

        skipped_matches += 1
        continue

    # =====================================================
    # PREDICT
    # =====================================================

    try:

        prediction = predict_match(
            red,
            blue
        )

    except Exception as e:

        print(
            f"Prediction failed: {e}"
        )

        skipped_matches += 1
        continue

    predicted_winner = prediction[
        "winner"
    ].lower()

    actual_winner = match[
        "winner"
    ].lower()

    predicted_red = prediction[
        "red_score"
    ]

    predicted_blue = prediction[
        "blue_score"
    ]

    actual_red = match[
        "red_score"
    ]

    actual_blue = match[
        "blue_score"
    ]

    # =====================================================
    # ACCURACY
    # =====================================================

    total_matches += 1

    if predicted_winner == actual_winner:

        correct_winners += 1

    score_error = (
        abs(predicted_red - actual_red)
        +
        abs(predicted_blue - actual_blue)
    ) / 2

    total_score_error += score_error

# =========================================================
# FINAL RESULTS
# =========================================================

if total_matches == 0:

    print("\nNo usable matches.\n")

    exit()

winner_accuracy = (
    correct_winners / total_matches
) * 100

average_score_error = (
    total_score_error / total_matches
)

# =========================================================
# OUTPUT
# =========================================================

print("\n" + "=" * 80)

print("MODEL VALIDATION RESULTS\n")

print(
    f"Matches Evaluated: "
    f"{total_matches}"
)

print(
    f"Matches Skipped: "
    f"{skipped_matches}"
)

print(
    f"\nWinner Accuracy: "
    f"{winner_accuracy:.2f}%"
)

print(
    f"Average Score Error: "
    f"{average_score_error:.2f}"
)

print("\n" + "=" * 80 + "\n")