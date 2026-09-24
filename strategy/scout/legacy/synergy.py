def calculate_alliance_synergy(alliance):

    score = 0

    trench_count = 0
    bump_count = 0

    long_cycle_count = 0
    congestion_sensitive = 0

    defender_count = 0

    for robot in alliance:

        route = robot.get("preferred_route", "unknown")

        if route == "trench":
            trench_count += 1

        if route == "bump":
            bump_count += 1

        if robot.get("cycle_style") == "long_cycles":
            long_cycle_count += 1

        if robot.get("congestion_susceptibility") == "high":
            congestion_sensitive += 1

        if robot.get("role") == "defender":
            defender_count += 1

    # =====================================================
    # PENALTIES
    # =====================================================

    if trench_count >= 3:
        score -= 20

    elif trench_count == 2:
        score -= 8

    if long_cycle_count >= 3:
        score -= 10

    if congestion_sensitive >= 2:
        score -= 12

    # =====================================================
    # BONUSES
    # =====================================================

    if trench_count > 0 and bump_count > 0:
        score += 10

    if defender_count >= 1:
        score += 5

    return score
