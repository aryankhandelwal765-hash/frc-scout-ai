# =========================================================
# ALLIANCE SYNERGY ENGINE
# =========================================================

import pandas as pd

# =========================================================
# HELPERS
# =========================================================

# =========================================================
# VALUE MAP
# =========================================================

VALUE_MAP = {
    "high": 8,
    "medium": 5,
    "low": 2,
    "unknown": 5
}

def numeric_trait(value):

    value = safe_string(value)

    return VALUE_MAP.get(value, 5)

def average(values):

    if len(values) == 0:
        return 0

    return sum(values) / len(values)


def safe_string(value):

    if value is None:
        return "unknown"

    return str(value).strip().lower()


# =========================================================
# ROLE BALANCE
# =========================================================

def calculate_role_balance(robots):

    roles = [
        safe_string(robot.get("role"))
        for robot in robots
    ]

    primary_count = roles.count("primary_scorer")

    hybrid_count = roles.count("hybrid")

    defender_count = roles.count("defender")

    # =====================================================
    # IDEAL:
    # 2 scorers + 1 hybrid/defender
    # =====================================================

    score = 7

    if primary_count == 3:
        score -= 2

    if defender_count >= 1:
        score += 1

    if hybrid_count >= 1:
        score += 1

    return max(1, min(10, round(score, 2)))


# =========================================================
# TRAFFIC RISK
# =========================================================

# =========================================================
# TRAFFIC RISK
# =========================================================

def calculate_traffic_risk(robots):

    congestion_scores = []

    throughput_scores = []

    for robot in robots:

        congestion_value = numeric_trait(
            robot.get("congestion_susceptibility")
        )

        congestion_scores.append(
            congestion_value
        )

        throughput_scores.append(
            float(
                robot.get(
                    "throughput_score",
                    5
                )
            )
        )

    avg_congestion = average(
        congestion_scores
    )

    avg_throughput = average(
        throughput_scores
    )

    # =====================================================
    # HIGH THROUGHPUT + HIGH CONGESTION
    # = traffic problems
    # =====================================================

    risk = (
        avg_congestion * 0.6 +
        (avg_throughput / 2) * 0.4
    )

    return max(
        1,
        min(
            10,
            round(risk, 2)
        )
    )

# =========================================================
# DEFENSIVE STABILITY
# =========================================================

def calculate_defensive_stability(robots):

    survivability_scores = [
        robot.get("survivability_score", 5)
        for robot in robots
    ]

    reliability_scores = [
        robot.get("reliability_score", 5)
        for robot in robots
    ]

    score = (
        average(survivability_scores) * 0.6 +
        average(reliability_scores) * 0.4
    )

    return max(1, min(10, round(score, 2)))


# =========================================================
# OFFENSIVE POWER
# =========================================================

def calculate_offensive_power(robots):

    throughput_scores = [
        robot.get("throughput_score", 5)
        for robot in robots
    ]

    offensive_ceiling_scores = [
        robot.get("offensive_ceiling", 5)
        for robot in robots
    ]

    score = (
        average(throughput_scores) * 0.45 +
        average(offensive_ceiling_scores) * 0.55
    )

    return max(1, min(10, round(score, 2)))


# =========================================================
# ROUTE DIVERSITY
# =========================================================

def calculate_route_diversity(robots):

    route_types = []

    for robot in robots:

        route = safe_string(
            robot.get("route_flexibility")
        )

        route_types.append(route)

    unique_routes = len(set(route_types))

    if unique_routes >= 3:
        return 9

    if unique_routes == 2:
        return 7

    return 4


# =========================================================
# OVERALL SYNERGY
# =========================================================

def calculate_overall_synergy(robots):

    offensive_power = calculate_offensive_power(
        robots
    )

    defensive_stability = (
        calculate_defensive_stability(
            robots
        )
    )

    role_balance = calculate_role_balance(
        robots
    )

    route_diversity = (
        calculate_route_diversity(
            robots
        )
    )

    traffic_risk = calculate_traffic_risk(
        robots
    )

    # =====================================================
    # TRAFFIC RISK IS BAD
    # =====================================================

    traffic_inverse = 11 - traffic_risk

    score = (
        offensive_power * 0.35 +
        defensive_stability * 0.25 +
        role_balance * 0.15 +
        route_diversity * 0.10 +
        traffic_inverse * 0.15
    )

    return max(1, min(10, round(score, 2)))


# =========================================================
# FULL SYNERGY ANALYSIS
# =========================================================

def analyze_alliance(robots):

    offensive_power = calculate_offensive_power(
        robots
    )

    defensive_stability = (
        calculate_defensive_stability(
            robots
        )
    )

    role_balance = calculate_role_balance(
        robots
    )

    route_diversity = (
        calculate_route_diversity(
            robots
        )
    )

    traffic_risk = calculate_traffic_risk(
        robots
    )

    overall_synergy = (
        calculate_overall_synergy(
            robots
        )
    )

    predicted_score = round(
        (
                offensive_power * 8
        ) +
        (
                defensive_stability * 4
        ) +
        (
                overall_synergy * 5
        ),
        1
    )

    return {

        "offensive_power":
            offensive_power,

        "defensive_stability":
            defensive_stability,

        "role_balance":
            role_balance,

        "route_diversity":
            route_diversity,

        "traffic_risk":
            traffic_risk,

        "overall_synergy":
            overall_synergy,

        "predicted_score":
            predicted_score
    }