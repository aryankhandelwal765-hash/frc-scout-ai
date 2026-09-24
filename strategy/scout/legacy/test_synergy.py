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

CSV_PATH = os.path.join(SCOUTING_DATA_DIR, "scouting_with_metrics_20260508_123118.csv")

ALLIANCE = [
    2056,
    581,
    1787
]

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CSV_PATH)

df["Team Number"] = pd.to_numeric(
    df["Team Number"],
    errors="coerce"
)

# =========================================================
# EXTRACT ROBOTS
# =========================================================

robots = []

for team in ALLIANCE:

    matching = df[
        df["Team Number"] == team
    ]

    if matching.empty:

        print(f"ERROR: Team {team} not found in CSV")

        continue

    robot = matching.iloc[0]

    robots.append(robot)

# =========================================================
# VALIDATION
# =========================================================

if len(robots) != 3:

    print("\nAlliance incomplete. Exiting.\n")

    exit()

# =========================================================
# RUN ANALYSIS
# =========================================================

results = analyze_alliance(robots)

# =========================================================
# OUTPUT
# =========================================================

print("\n" + "=" * 80)

print("ALLIANCE\n")

for robot in robots:

    print(
        f"{int(robot['Team Number'])} - "
        f"{robot['Team Name']}"
    )

print("\n" + "=" * 80)

print("ALLIANCE ANALYSIS\n")

for key, value in results.items():

    print(f"{key}: {value}")

print("\n" + "=" * 80 + "\n")