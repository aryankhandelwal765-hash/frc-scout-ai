import time
import joblib
import pandas as pd

from pathlib import Path

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score
)

from sklearn.preprocessing import (
    StandardScaler
)

from scout.config import (
    MODELS_DIR,
    PROCESSED_DIR,
    MATCHUP_MODEL_DIR
)

from scout.engine.model_features import (
    MODEL_FEATURES
)

# =========================================================
# TIMER
# =========================================================

start_time = time.time()

# =========================================================
# CONFIG
# =========================================================

DATASET_FILE = (
    PROCESSED_DIR
    /
    "hybrid_matchup_dataset.csv"
)

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading matchup dataset...\n")

df = pd.read_csv(
    DATASET_FILE
)

print(
    f"Loaded {len(df)} matchup rows.\n"
)

# =========================================================
# SPLIT BY DIVISION
# =========================================================

worlds_df = df[

    df["is_worlds"] == True
]

train_df = df[

    df["is_worlds"] == False
]

print(
    f"Training rows: "
    f"{len(train_df)}"
)

print(
    f"Testing rows (Worlds): "
    f"{len(worlds_df)}\n"
)
# =========================================================
# FEATURES
# =========================================================

X_train = train_df[
    MODEL_FEATURES
]

X_test = worlds_df[
    MODEL_FEATURES
]

# =========================================================
# TARGETS
# =========================================================

y_train_diff = train_df[
    "score_diff"
]

y_test_diff = worlds_df[
    "score_diff"
]

y_train_total = train_df[
    "total_score"
]

y_test_total = worlds_df[
    "total_score"
]

y_train_win = train_df[
    "red_win"
]

y_test_win = worlds_df[
    "red_win"
]

# =========================================================
# SCALE
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =========================================================
# SCORE DIFFERENTIAL MODEL
# =========================================================

print(
    "Training score differential model..."
)

score_model = RandomForestRegressor(

    n_estimators=400,

    max_depth=14,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)

score_model.fit(

    X_train_scaled,

    y_train_diff
)

# =========================================================
# TOTAL SCORE MODEL
# =========================================================

print(
    "Training total score model..."
)

total_model = RandomForestRegressor(

    n_estimators=400,

    max_depth=14,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)

total_model.fit(

    X_train_scaled,

    y_train_total
)

# =========================================================
# WIN CLASSIFIER
# =========================================================

print(
    "Training win classifier..."
)

win_model = RandomForestClassifier(

    n_estimators=400,

    max_depth=14,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)

win_model.fit(

    X_train_scaled,

    y_train_win
)

# =========================================================
# EVALUATION
# =========================================================

score_predictions = (
    score_model.predict(
        X_test_scaled
    )
)

score_mae = mean_absolute_error(

    y_test_diff,

    score_predictions
)

total_predictions = (
    total_model.predict(
        X_test_scaled
    )
)

total_mae = mean_absolute_error(

    y_test_total,

    total_predictions
)

win_predictions = (
    win_model.predict(
        X_test_scaled
    )
)

win_accuracy = accuracy_score(

    y_test_win,

    win_predictions
)

# =========================================================
# IQM HELPER
# =========================================================

def iqm(values):

    values = sorted(values)

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
# IQM METRICS
# =========================================================

margin_errors = [

    abs(a - p)

    for a, p in zip(
        y_test_diff,
        score_predictions
    )
]

total_errors = [

    abs(a - p)

    for a, p in zip(
        y_test_total,
        total_predictions
    )
]

margin_iqm = iqm(
    margin_errors
)

total_iqm = iqm(
    total_errors
)

# =========================================================
# OUTPUT
# =========================================================

print("\n" + "=" * 80)

print(
    "\nGENERALIZATION RESULTS "
    "(Train: District + Regional Events | Test: Worlds)\n"
)

print(
    f"Margin MAE: "
    f"{score_mae:.2f}"
)

print(
    f"Margin IQM Error: "
    f"{margin_iqm:.2f}"
)

print(
    f"Total Score MAE: "
    f"{total_mae:.2f}"
)

print(
    f"Total Score IQM Error: "
    f"{total_iqm:.2f}"
)

print(
    f"Winner Accuracy: "
    f"{win_accuracy * 100:.2f}%"
)

print("\n" + "=" * 80)

# =========================================================
# FEATURE IMPORTANCES
# =========================================================

print("\nFeature Importances:\n")

for feature, importance in zip(

    MODEL_FEATURES,

    score_model.feature_importances_
):

    print(
        f"{feature}: "
        f"{importance:.3f}"
    )

# =========================================================
# SAVE
# =========================================================

MODELS_DIR.mkdir(

    parents=True,

    exist_ok=True
)

joblib.dump(

    score_model,

    MODELS_DIR /
    "matchup_score_model.pkl"
)

joblib.dump(

    total_model,

    MODELS_DIR /
    "matchup_total_model.pkl"
)

joblib.dump(

    win_model,

    MODELS_DIR /
    "matchup_win_model.pkl"
)

joblib.dump(

    scaler,

    MODELS_DIR /
    "matchup_scaler.pkl"
)

joblib.dump(

    MODEL_FEATURES,

    MATCHUP_MODEL_DIR
    /
    "feature_columns.pkl"
)

print(
    "\nSaved matchup models."
)

# =========================================================
# TIMER OUTPUT
# =========================================================

elapsed = (
    time.time()
    -
    start_time
)

print(
    f"\nRuntime: "
    f"{elapsed:.2f} seconds\n"
)