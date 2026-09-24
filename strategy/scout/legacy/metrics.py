LEVEL_MAP = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "unknown": 2
}

SPEED_MAP = {
    "slow": 1,
    "medium": 2,
    "fast": 3,
    "unknown": 2
}

ROLE_MAP = {
    "defender": 1,
    "secondary": 2,
    "hybrid": 2.5,
    "primary_scorer": 3,
    "unknown": 2
}


def calculate_robot_metrics(robot):

    throughput = (
        SPEED_MAP.get(robot["cycle_speed"], 2) * 2
        + LEVEL_MAP.get(robot["intake_quality"], 2)
        + LEVEL_MAP.get(robot["traffic_handling"], 2)
        + LEVEL_MAP.get(robot["short_cycle_capability"], 2)
    )

    defense_stability = (
        LEVEL_MAP.get(robot["defense_resistance"], 2) * 2
        + LEVEL_MAP.get(robot["driver_coordination"], 2)
        - LEVEL_MAP.get(robot["congestion_susceptibility"], 2)
    )

    routing = (
        LEVEL_MAP.get(robot["route_flexibility"], 2)
        + LEVEL_MAP.get(robot["traffic_handling"], 2)
    )

    reliability = (
        LEVEL_MAP.get(robot["mechanical_reliability"], 2)
        + LEVEL_MAP.get(robot["targeting_reliability"], 2)
        + LEVEL_MAP.get(robot["driver_confidence"], 2)
    )

    overall = (
        throughput * 0.35
        + defense_stability * 0.25
        + routing * 0.20
        + reliability * 0.20
    )

    return {
        "throughput": round(throughput, 2),
        "defense_stability": round(defense_stability, 2),
        "routing": round(routing, 2),
        "reliability": round(reliability, 2),
        "overall": round(overall, 2)
    }