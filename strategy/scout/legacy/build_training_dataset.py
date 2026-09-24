import json
import pandas as pd

from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

MATCH_FILE = (
    "data/raw_matches/"
    "2026_worlds_all_matches.json"
)

TEAM_DATA_FILE = (
    "data/statbotics/"
    "worlds_2026_team_data.json"
)

OUTPUT_FILE = (
    "data/processed/"
    "training_dataset.csv"
)

Path(
    "data/processed"
).mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading matches...\n")

with open(MATCH_FILE, "r") as f:

    matches = json.load(f)

print(
    f"Loaded {len(matches)} matches."
)

print("\nLoading team EPA data...\n")

with open(TEAM_DATA_FILE, "r") as f:

    team_data = json.load(f)

print(
    f"Loaded {len(team_data)} teams."
)

# =========================================================
# HELPERS
# =========================================================

def get_team(team_number):

    return team_data.get(
        str(team_number)
    )

def alliance_features(team_list):

    robots = []

    for team in team_list:

        robot = get_team(team)

        if robot is None:

            return None

        robots.append(robot)

    epa_values = [
        r["epa"]
        for r in robots
    ]

    auto_values = [
        r["auto_epa"]
        for r in robots
    ]

    teleop_values = [
        r["teleop_epa"]
        for r in robots
    ]

    endgame_values = [
        r["endgame_epa"]
        for r in robots
    ]

    winrates = [
        r["winrate"]
        for r in robots
    ]

    return {

        "epa_sum":
            sum(epa_values),



        "epa_max":
            max(epa_values),

        "auto_epa_sum":
            sum(auto_values),

        "teleop_epa_sum":
            sum(teleop_values),

        "endgame_epa_sum":
            sum(endgame_values),

        "winrate_avg":
            sum(winrates) / 3
    }

# =========================================================
# BUILD DATASET
# =========================================================

rows = []

print("\nBuilding alliance dataset...\n")

for match in matches:

    red_features = alliance_features(
        match["red_teams"]
    )

    blue_features = alliance_features(
        match["blue_teams"]
    )

    if (
        red_features is None
        or
        blue_features is None
    ):
        continue

    # =====================================================
    # RED ROW
    # =====================================================

    red_row = red_features.copy()

    red_row["alliance"] = "red"

    red_row["actual_score"] = (
        match["red_score"]
    )

    red_row["won"] = int(
        match["winner"].lower()
        == "red"
    )

    red_row["comp_level"] = (
        match["comp_level"]
    )

    rows.append(red_row)

    # =====================================================
    # BLUE ROW
    # =====================================================

    blue_row = blue_features.copy()

    blue_row["alliance"] = "blue"

    blue_row["actual_score"] = (
        match["blue_score"]
    )

    blue_row["won"] = int(
        match["winner"].lower()
        == "blue"
    )

    blue_row["comp_level"] = (
        match["comp_level"]
    )

    rows.append(blue_row)

# =========================================================
# SAVE
# =========================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nGenerated "
    f"{len(df)} alliance samples."
)

print(
    f"\nSaved:\n"
    f"{OUTPUT_FILE}\n"
)