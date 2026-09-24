import os

# Directory holding the raw scouting CSV exports. Override with the
# SCOUTING_DATA_DIR env var; defaults to ~/Downloads.
SCOUTING_DATA_DIR = os.environ.get(
    "SCOUTING_DATA_DIR", os.path.expanduser("~/Downloads")
)

import pandas as pd

from scout.legacy.alliance_synergy import analyze_alliance

# =========================================================
# CONFIG
# =========================================================

CSV_PATH = os.path.join(SCOUTING_DATA_DIR, "scouting_structured_20260508_053231.csv")


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CSV_PATH)

# =========================================================
# PICK ALLIANCE
# =========================================================

ALLIANCE = [
    2056,
    581,
    1787
]

# =========================================================
# EXTRACT ROBOTS
# =========================================================


robots = []

for team_number in ALLIANCE:

    robot = df[
        df["Team Number"] == team_number
    ].iloc[0]

    robots.append(robot)

# =========================================================
# RUN ANALYSIS
# =========================================================

results = analyze_alliance(robots)

# =========================================================
# PRINT TEAMS
# =========================================================

print("\n" + "=" * 80)

print("ALLIANCE\n")

for robot in robots:

    print(
        f"{robot['Team Number']} - "
        f"{robot['Team Name']}"
    )

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 80)

print("ALLIANCE ANALYSIS\n")

for key, value in results.items():

    print(f"{key}: {value}")

print("\n" + "=" * 80 + "\n")