import os

# Directory holding the raw scouting CSV exports. Override with the
# SCOUTING_DATA_DIR env var; defaults to ~/Downloads.
SCOUTING_DATA_DIR = os.environ.get(
    "SCOUTING_DATA_DIR", os.path.expanduser("~/Downloads")
)

import pandas as pd

from metric_engine import calculate_all_metrics

# =========================================================
# CONFIG
# =========================================================

CSV_PATH = os.path.join(SCOUTING_DATA_DIR, "scouting_with_metrics_20260508_123118.csv")

TEAM_NUMBER = 2056

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CSV_PATH)

df["Team Number"] = pd.to_numeric(
    df["Team Number"],
    errors="coerce"
)

# =========================================================
# FIND TEAM
# =========================================================

matching = df[
    df["Team Number"] == TEAM_NUMBER
]

if matching.empty:

    print(f"ERROR: Team {TEAM_NUMBER} not found.")

    exit()

robot = matching.iloc[0]

# =========================================================
# CALCULATE METRICS
# =========================================================

metrics = calculate_all_metrics(robot)

# =========================================================
# OUTPUT
# =========================================================

print("\n" + "=" * 50)

print(f"TEAM: {robot['Team Name']} ({TEAM_NUMBER})")

print("=" * 50)

print("\nRAW EXTRACTION:\n")

fields = [
    "cycle_speed",
    "defense_resistance",
    "congestion_susceptibility",
    "route_flexibility",
    "driver_confidence",
    "driver_coordination",
    "intake_quality",
    "targeting_reliability",
    "mechanical_reliability",
    "role"
]

for field in fields:

    print(f"{field}: {robot.get(field)}")

print("\nTACTICAL METRICS:\n")

for key, value in metrics.items():

    print(f"{key}: {value}")

print("\n" + "=" * 50 + "\n")