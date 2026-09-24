import joblib
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.ensemble import (
    RandomForestRegressor
)

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = (
    "data/processed/"
    "training_dataset.csv"
)

MODEL_OUTPUT = (
    "models/match_score_model.pkl"
)

SCALER_OUTPUT = (
    "models/match_feature_scaler.pkl"
)

# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading training dataset...\n")

df = pd.read_csv(DATASET_PATH)

print(
    f"Loaded {len(df)} alliance samples.\n"
)

# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [

    "epa_sum",

    "epa_max",

    "auto_epa_sum",

    "teleop_epa_sum",

    "endgame_epa_sum",

    "winrate_avg"
]

X = df[
    FEATURE_COLUMNS
]

# =========================================================
# TARGETS
# =========================================================

y_score = df[
    "actual_score"
]

y_win = df[
    "won"
]

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y_score,
        test_size=0.2,
        random_state=42
    )
)

# =========================================================
# SCALE FEATURES
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# =========================================================
# MODEL
# =========================================================

print("Training score regression model...\n")
# =========================================================
# RIDGE MODEL
# =========================================================

print(
    "Training Ridge model..."
)

ridge_model = Ridge(
    alpha=5.0
)

ridge_model.fit(
    X_train_scaled,
    y_train
)

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

print(
    "Training Random Forest model..."
)

forest_model = RandomForestRegressor(

    n_estimators=300,

    max_depth=12,

    min_samples_split=4,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1
)

forest_model.fit(
    X_train_scaled,
    y_train
)

# =========================================================
# ENSEMBLE PREDICTION
# =========================================================

ridge_predictions = (
    ridge_model.predict(
        X_test_scaled
    )
)

forest_predictions = (
    forest_model.predict(
        X_test_scaled
    )
)

score_predictions = (

    ridge_predictions * 0.4

    +

    forest_predictions * 0.6
)
score_mae = mean_absolute_error(
    y_test,
    score_predictions
)

# =========================================================
# WIN PREDICTION
# =========================================================

median_score = df[
    "actual_score"
].median()

predicted_wins = [
    int(score > median_score)
    for score in score_predictions
]

actual_wins = y_win.loc[
    y_test.index
]

win_accuracy = accuracy_score(
    actual_wins,
    predicted_wins
)

# =========================================================
# OUTPUT
# =========================================================

print("=" * 80)

print("\nMODEL RESULTS\n")

print(
    f"Score MAE: "
    f"{score_mae:.2f}"
)

print(
    f"Win Accuracy: "
    f"{win_accuracy * 100:.2f}%"
)

print("\n" + "=" * 80)
# =========================================================
# OUTPUT
# =========================================================

print("=" * 80)

print("\nMODEL RESULTS\n")

print(
    f"Score MAE: "
    f"{score_mae:.2f}"
)

print(
    f"Win Accuracy: "
    f"{win_accuracy * 100:.2f}%"
)

# =========================================================
# RIDGE WEIGHTS
# =========================================================

print("\nRidge Feature Weights:\n")

for feature, coef in zip(
    FEATURE_COLUMNS,
    ridge_model.coef_
):

    print(
        f"{feature}: "
        f"{coef:.3f}"
    )

print(
    f"\nRidge Intercept: "
    f"{ridge_model.intercept_:.3f}"
)

# =========================================================
# RANDOM FOREST IMPORTANCE
# =========================================================

print("\nForest Feature Importances:\n")

for feature, importance in zip(
    FEATURE_COLUMNS,
    forest_model.feature_importances_
):

    print(
        f"{feature}: "
        f"{importance:.3f}"
    )

print("\n" + "=" * 80)

# =========================================================
# SAVE MODELS
# =========================================================

joblib.dump(
    ridge_model,
    "models/ridge_model.pkl"
)

joblib.dump(
    forest_model,
    "models/forest_model.pkl"
)

joblib.dump(
    scaler,
    SCALER_OUTPUT
)

print(
    "\nSaved Ridge model:\n"
    "models/ridge_model.pkl"
)

print(
    "\nSaved Forest model:\n"
    "models/forest_model.pkl"
)

print(
    f"\nSaved scaler:\n"
    f"{SCALER_OUTPUT}\n"
)