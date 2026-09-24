import os

# Directory holding the raw scouting CSV exports. Override with the
# SCOUTING_DATA_DIR env var; defaults to ~/Downloads.
SCOUTING_DATA_DIR = os.environ.get(
    "SCOUTING_DATA_DIR", os.path.expanduser("~/Downloads")
)

import pandas as pd
from datetime import datetime

from metric_engine import calculate_all_metrics

# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = os.path.join(SCOUTING_DATA_DIR, "scouting_structured_20260508_053231.csv")

DOWNLOADS_FOLDER = SCOUTING_DATA_DIR

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_CSV = (
    f"{DOWNLOADS_FOLDER}/scouting_with_metrics_{TIMESTAMP}.csv"
)

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading structured scouting CSV...\n")

df = pd.read_csv(INPUT_CSV)

print(f"Loaded {len(df)} teams.\n")

# =========================================================
# CALCULATE METRICS
# =========================================================

metric_rows = []

print("Calculating tactical metrics...\n")

for _, row in df.iterrows():

    metrics = calculate_all_metrics(row)

    metric_rows.append(metrics)

# =========================================================
# CREATE METRICS DATAFRAME
# =========================================================

metrics_df = pd.DataFrame(metric_rows)

# =========================================================
# MERGE
# =========================================================

final_df = pd.concat(
    [
        df.reset_index(drop=True),
        metrics_df.reset_index(drop=True)
    ],
    axis=1
)

# =========================================================
# SORT BY OVERALL SCORE
# =========================================================

final_df = final_df.sort_values(
    by="overall_tactical_score",
    ascending=False
)

# =========================================================
# SAVE
# =========================================================

final_df.to_csv(
    OUTPUT_CSV,
    index=False
)

# =========================================================
# TOP TEAMS PREVIEW
# =========================================================

print("=" * 80)

print("TOP TACTICAL TEAMS\n")

preview_columns = [
    "Team Number",
    "Team Name",
    "Avg EPA",
    "throughput_score",
    "survivability_score",
    "reliability_score",
    "overall_tactical_score"
]

print(
    final_df[preview_columns]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 80)

print(f"\nSaved metric-enhanced CSV:\n")

print(OUTPUT_CSV)

print("\nDone.\n")