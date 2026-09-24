# =========================================================
# METRIC ENGINE V2
# =========================================================

import math

# =========================================================
# VALUE MAPS
# =========================================================

VALUE_MAP = {
    "high": 2.7,
    "medium": 1.9,
    "low": 1.0,
    "unknown": 1.5
}

ROLE_MAP = {
    "primary_scorer": 3.0,
    "secondary": 2.0,
    "secondary_scorer": 2.0,
    "hybrid": 2.0,
    "defender": 1.2,
    "unknown": 1.5
}

# =========================================================
# HELPERS
# =========================================================

def safe_string(value):

    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    if value == "nan":
        return "unknown"

    return value


def get_value(value):

    value = safe_string(value)

    return VALUE_MAP.get(value, 1.5)


def get_role_value(value):

    value = safe_string(value)

    return ROLE_MAP.get(value, 1.5)


def safe_epa(robot):

    try:
        return float(robot.get("Avg EPA", 0))
    except:
        return 0


def clamp_score(score):

    return max(1, min(10, round(score, 2)))


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_trait_score(raw_score):

    """
    Converts 1.0–2.7 weighted traits
    into more realistic 1–10 tactical scaling.
    """

    normalized = ((raw_score - 1.0) / 1.7) * 8 + 1

    # Slight nonlinear compression
    normalized = normalized ** 0.96

    return clamp_score(normalized)


# =========================================================
# THROUGHPUT
# =========================================================

def calculate_throughput(robot):

    cycle_speed = get_value(
        robot.get("cycle_speed")
    )

    intake = get_value(
        robot.get("intake_quality")
    )

    targeting = get_value(
        robot.get("targeting_reliability")
    )

    coordination = get_value(
        robot.get("driver_coordination")
    )

    raw_score = (
        cycle_speed * 0.35 +
        intake * 0.25 +
        targeting * 0.25 +
        coordination * 0.15
    )

    return normalize_trait_score(raw_score)


# =========================================================
# DEFENSIVE SURVIVABILITY
# =========================================================

def calculate_survivability(robot):

    defense = get_value(
        robot.get("defense_resistance")
    )

    congestion = get_value(
        robot.get("congestion_susceptibility")
    )

    # low congestion susceptibility is GOOD
    congestion_inverse = 3.7 - congestion

    flexibility = get_value(
        robot.get("route_flexibility")
    )

    mechanical = get_value(
        robot.get("mechanical_reliability")
    )

    raw_score = (
        defense * 0.45 +
        congestion_inverse * 0.25 +
        flexibility * 0.20 +
        mechanical * 0.10
    )

    return normalize_trait_score(raw_score)


# =========================================================
# RELIABILITY
# =========================================================

def calculate_reliability(robot):

    mechanical = get_value(
        robot.get("mechanical_reliability")
    )

    confidence = get_value(
        robot.get("driver_confidence")
    )

    coordination = get_value(
        robot.get("driver_coordination")
    )

    raw_score = (
        mechanical * 0.40 +
        confidence * 0.35 +
        coordination * 0.25
    )

    return normalize_trait_score(raw_score)


# =========================================================
# OFFENSIVE CEILING
# =========================================================

def calculate_offensive_ceiling(robot):

    throughput = calculate_throughput(robot)

    targeting = normalize_trait_score(
        get_value(
            robot.get("targeting_reliability")
        )
    )

    role = (
        get_role_value(
            robot.get("role")
        ) / 3
    ) * 10

    epa = safe_epa(robot)

    # Much more realistic EPA normalization
    normalized_epa = min(10, epa / 50)

    score = (
        throughput * 0.45 +
        targeting * 0.20 +
        role * 0.15 +
        normalized_epa * 0.20
    )

    return clamp_score(score)


# =========================================================
# PLAYOFF STABILITY
# =========================================================

def calculate_playoff_stability(robot):

    survivability = calculate_survivability(robot)

    reliability = calculate_reliability(robot)

    congestion = get_value(
        robot.get("congestion_susceptibility")
    )

    congestion_inverse = normalize_trait_score(
        3.7 - congestion
    )

    score = (
        survivability * 0.45 +
        reliability * 0.35 +
        congestion_inverse * 0.20
    )

    return clamp_score(score)


# =========================================================
# OVERALL TACTICAL SCORE
# =========================================================

def calculate_overall(robot):

    throughput = calculate_throughput(robot)

    survivability = calculate_survivability(robot)

    reliability = calculate_reliability(robot)

    offensive_ceiling = calculate_offensive_ceiling(robot)

    playoff_stability = calculate_playoff_stability(robot)

    epa = safe_epa(robot)

    normalized_epa = min(10, epa / 50)

    score = (
        throughput * 0.20 +
        survivability * 0.20 +
        reliability * 0.15 +
        offensive_ceiling * 0.25 +
        playoff_stability * 0.10 +
        normalized_epa * 0.10
    )

    return clamp_score(score)


# =========================================================
# FULL METRIC EXTRACTION
# =========================================================

def calculate_all_metrics(robot):

    return {

        "throughput_score":
            calculate_throughput(robot),

        "survivability_score":
            calculate_survivability(robot),

        "reliability_score":
            calculate_reliability(robot),

        "offensive_ceiling":
            calculate_offensive_ceiling(robot),

        "playoff_stability":
            calculate_playoff_stability(robot),

        "overall_tactical_score":
            calculate_overall(robot)
    }