import json
import pandas as pd

from pathlib import Path

from scout.config import (

    RAW_MATCHES_DIR,

    STATBOTICS_DIR,

    SCOUTING_DIR,

    PROCESSED_DIR
)

from scout.engine.feature_engineering import (

    build_statbotics_lookup,

    build_scouting_lookup,

    build_matchup_features
)

# =========================================================
# CONFIG
# =========================================================

MATCH_FILE = (
    RAW_MATCHES_DIR
    /
    "2026_all_matches.json"
)

TEAM_DATA_FILE = (
    STATBOTICS_DIR
    /
    "all_2026_team_data.json"
)

SCOUTING_FILE = (
    SCOUTING_DIR
    /
    "team_scouting_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    /
    "hybrid_matchup_dataset.csv"
)

# =========================================================
# LOAD MATCHES
# =========================================================

print("\nLoading matches...\n")

with open(MATCH_FILE, "r") as f:

    matches = json.load(f)

print(
    f"Loaded "
    f"{len(matches)} matches.\n"
)

# =========================================================
# LOAD TEAM DATA
# =========================================================

print("Loading Statbotics data...\n")

with open(TEAM_DATA_FILE, "r") as f:

    team_data = json.load(f)

print(
    f"Loaded "
    f"{len(team_data)} teams.\n"
)

# =========================================================
# LOAD SCOUTING
# =========================================================

print("Loading scouting data...\n")

scouting_df = pd.read_csv(
    SCOUTING_FILE
)

print(
    f"Loaded "
    f"{len(scouting_df)} scouting rows.\n"
)

# =========================================================
# BUILD LOOKUPS
# =========================================================

statbotics_lookup = (
    build_statbotics_lookup(
        team_data
    )
)

scouting_lookup = (
    build_scouting_lookup(
        scouting_df
    )
)

# =========================================================
# BUILD DATASET
# =========================================================

print("Building hybrid dataset...\n")

rows = []

for match in matches:

    try:


        red_teams = match[
            "red_teams"
        ]

        blue_teams = match[
            "blue_teams"
        ]

        red_score = match[
            "red_score"
        ]

        blue_score = match[
            "blue_score"
        ]

    except Exception as e:

        print("\nFAILED MATCH\n")

        print(match)

        print("\nERROR:\n")

        print(e)

        continue

    # =====================================================
    # CENTRALIZED FEATURE GENERATION
    # =====================================================

    features = build_matchup_features(

        red_teams,
        blue_teams,

        statbotics_lookup,

        scouting_lookup
    )

    # =====================================================
    # BUILD ROW
    # =====================================================

    row = {
        "match_key":
            match["match_key"],

        "event_key":
            match["event_key"],

        "is_worlds":
            (
                    match["event_key"]
                    in
                    [

                        "2026hop",
                        "2026new",
                        "2026cur",
                        "2026arc",
                        "2026gal",
                        "2026dal",
                        "2026joh",
                        "2026mil"
                    ]
            ),


        "red_1":
            red_teams[0],

        "red_2":
            red_teams[1],

        "red_3":
            red_teams[2],

        "blue_1":
            blue_teams[0],

        "blue_2":
            blue_teams[1],

        "blue_3":
            blue_teams[2],

        **features,

        "red_score":
            red_score,

        "blue_score":
            blue_score,

        "score_diff":
            red_score - blue_score,

        "total_score":
            red_score + blue_score,

        "red_win":
            int(red_score > blue_score)
    }

    rows.append(row)

# =========================================================
# SAVE
# =========================================================

df = pd.DataFrame(rows)

PROCESSED_DIR.mkdir(

    parents=True,

    exist_ok=True
)

df.to_csv(

    OUTPUT_FILE,

    index=False
)

# =========================================================
# OUTPUT
# =========================================================

print(
    f"\nGenerated "
    f"{len(df)} hybrid rows."
)

print(
    f"\nSaved:\n"
    f"{OUTPUT_FILE}\n"
)

print("\nHYBRID DATASET COLUMNS\n")

for col in df.columns:

    print(col)