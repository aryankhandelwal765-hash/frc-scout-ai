import json
import pandas as pd

from scout.config import (

    RAW_MATCHES_DIR,

    STATBOTICS_DIR,

    PROCESSED_DIR
)

from scout.engine.feature_engineering import (

    build_statbotics_lookup,

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

OUTPUT_FILE = (
    PROCESSED_DIR
    /
    "matchup_dataset.csv"
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
# BUILD LOOKUP
# =========================================================

statbotics_lookup = (
    build_statbotics_lookup(
        team_data
    )
)

# =========================================================
# BUILD DATASET
# =========================================================

print("Building matchup dataset...\n")

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
    # BUILD FEATURES
    # =====================================================

    features = build_matchup_features(

        red_teams,
        blue_teams,

        statbotics_lookup,

        scouting_lookup={}
    )

    # =====================================================
    # BUILD ROW
    # =====================================================

    row = {

        # =================================================
        # TEAM IDS
        # =================================================
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

        # =================================================
        # FEATURES
        # =================================================

        **features,

        # =================================================
        # TARGETS
        # =================================================

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
    f"{len(df)} matchup rows."
)

print(
    f"\nSaved:\n"
    f"{OUTPUT_FILE}\n"
)

print("\nDATASET COLUMNS\n")

for col in df.columns:

    print(col)