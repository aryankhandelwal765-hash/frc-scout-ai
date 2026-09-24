import pandas as pd

from scout.config import (
    SCOUTING_DIR
)

# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = (
    SCOUTING_DIR
    /
    "vision_2026hop.csv"
)

OUTPUT_FILE = (
    SCOUTING_DIR
    /
    "team_scouting_features.csv"
)

# =========================================================
# LOAD
# =========================================================

print("\nLoading scouting CSV...\n")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Loaded "
    f"{len(df)} rows.\n"
)

# =========================================================
# CLEAN
# =========================================================

# =========================================================
# CLEAN
# =========================================================

df.columns = [

    c.strip().lower()

    for c in df.columns
]

print("\nSCOUTING COLUMNS:\n")

for c in df.columns:

    print(c)

# =========================================================
# IQM HELPER
# =========================================================

def iqm(series):

    values = sorted(

        series.dropna().tolist()
    )

    if len(values) == 0:

        return 0

    if len(values) < 4:

        return sum(values) / len(values)

    lower = int(
        len(values) * 0.25
    )

    upper = int(
        len(values) * 0.75
    )

    trimmed = values[
        lower:upper
    ]

    if len(trimmed) == 0:

        return sum(values) / len(values)

    return (
        sum(trimmed)
        /
        len(trimmed)
    )

# =========================================================
# GROUP
# =========================================================

grouped = df.groupby(
    "team"
)

rows = []

# =========================================================
# BUILD FEATURES
# =========================================================

for team, group in grouped:

    matches = len(group)

    row = {

        "team":
            int(team),

        "matches":
            matches,

        "scouting_confidence":
            min(
                matches / 8,
                1.0
            ),

        "avg_auto_fuel":
            iqm(
                group.get(
                    "auto_fuel",
                    pd.Series(dtype=float)
                )
            ),

        "climb_success_rate":
            (
                group.get(
                    "climb",
                    pd.Series(dtype=float)
                ) == 1
            ).mean(),

        "failed_climb_rate":
            (
                group.get(
                    "failed_climb",
                    pd.Series(dtype=float)
                ) == 1
            ).mean(),

        "defense_frequency":
            (
                group.get(
                    "defense",
                    pd.Series(dtype=float)
                ) == 1
            ).mean(),

        "trench_usage_rate":
            (
                group.get(
                    "tele_trench",
                    pd.Series(dtype=float)
                ) == 1
            ).mean(),

        "bump_usage_rate":
            (
                group.get(
                    "tele_bump",
                    pd.Series(dtype=float)
                ) == 1
            ).mean(),

        "passing_usage_rate":
            (
                group.get(
                    "tele_passing",
                    pd.Series(dtype=float)
                ) == 1
            ).mean(),

        "avg_hp_accuracy":
            iqm(
                group.get(
                    "hp_accuracy",
                    pd.Series(dtype=float)
                )
            ),

        "avg_driver_skill":
            iqm(
                group.get(
                    "driver_skill",
                    pd.Series(dtype=float)
                )
            ),

        "avg_reliability":
            iqm(
                group.get(
                    "reliability",
                    pd.Series(dtype=float)
                )
            )
    }

    rows.append(row)

# =========================================================
# VISION-DERIVED FEATURES (vision)
# =========================================================
# These come from broadcast video via vision/match_pipeline/export_to_strategy.py, which
# only exports metrics whose SPLIT-HALF RELIABILITY clears 0.55 -- i.e. metrics that
# describe the TEAM rather than the individual match. Anything noisier is withheld
# there on purpose, so if a column below is missing it is missing deliberately.

for row, (team, group) in zip(rows, grouped):

    row["avg_teleop_fuel"] = iqm(
        group.get("teleop_fuel", pd.Series(dtype=float))
    )

    row["avg_total_fuel"] = iqm(
        group.get("total_fuel", pd.Series(dtype=float))
    )

    row["avg_bump_visits"] = iqm(
        group.get("tele_bump_visits", pd.Series(dtype=float))
    )

    # what fraction of this team's matches had a CONFIDENT identity (bumper number
    # read, or resolved by elimination) rather than an unverified station guess
    src = group.get("identity_src", pd.Series(dtype=object))
    src = src.dropna() if hasattr(src, "dropna") else src
    row["identity_confidence"] = (
        float((src.isin(["read", "elim"])).mean()) if len(src) else 0.0
    )

# =========================================================
# HONEST GAP REPORT
# =========================================================
# iqm() returns 0 for an absent column, which reads as "the team never did that"
# rather than "nobody measured it". Say which features are that kind of zero.

_EXPECTED = [
    "auto_fuel", "climb", "failed_climb", "defense", "tele_bump", "tele_trench",
    "tele_passing", "hp_accuracy", "reliability", "driver_skill",
]
_absent = [c for c in _EXPECTED if c not in df.columns]

if _absent:
    print("\nCOLUMNS ABSENT FROM THE INPUT — their features are 0.0 because")
    print("nothing measured them, NOT because teams scored zero:")
    for c in _absent:
        print(f"   {c}")
    print("Do not read those aggregates as evidence.\n")

# =========================================================
# SAVE
# =========================================================

features_df = pd.DataFrame(rows)

SCOUTING_DIR.mkdir(

    parents=True,

    exist_ok=True
)

features_df.to_csv(

    OUTPUT_FILE,

    index=False
)

# =========================================================
# OUTPUT
# =========================================================

print(
    f"\nGenerated "
    f"{len(features_df)} teams."
)

print(
    f"\nSaved:\n"
    f"{OUTPUT_FILE}\n"
)

print(features_df.head())