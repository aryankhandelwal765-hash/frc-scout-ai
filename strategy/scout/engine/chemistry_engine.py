"""Chemistry engine — alliance synergy from EPA-derived archetypes.

Adapts the original alliance-synergy logic (removed in the public release) — role
balance, offensive power, redundancy risk — onto the EPA archetypes produced by
archetype_engine, since scouting-trait data is unavailable.

Reads the stats dicts from statbotics_loader; archetypes are computed internally
(or can be passed in to avoid recomputation).
"""

from scout.engine.archetype_engine import classify_alliance


def _clamp(value: float, lo: float = 1.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, round(value, 2)))


def analyze_chemistry(stats_list: list[dict], archetypes: list[dict] | None = None) -> dict:
    """Score how well a 3-team alliance's pieces fit together.

    Components (each 1-10):
    - phase_coverage: do the teams collectively cover auto / teleop / endgame?
    - overlap_penalty risk: multiple teams crowding the same dominant phase
      (esp. teleop) means contested game pieces and wasted redundancy.
    - top_heaviness: is the alliance's output concentrated in one robot
      (fragile) or spread across all three (resilient)?
    - endgame_strength: does the alliance have real climb/endgame capacity?
    """
    if archetypes is None:
        archetypes = classify_alliance(stats_list)

    if not stats_list:
        return {"overall_chemistry": 0.0, "notes": ["No teams provided."]}

    epas = [max(s.get("epa", 0.0), 0.0) for s in stats_list]
    epa_sum = sum(epas) or 1.0

    # --- Phase coverage: distinct dominant phases across the alliance -----------
    dominant_phases = [a["dominant_phase"] for a in archetypes]
    distinct_phases = {p for p in dominant_phases if p != "none"}
    phase_coverage = _clamp(3.0 + 2.5 * len(distinct_phases))  # 1 phase->5.5, 3->10

    # --- Scoring overlap: teams stacked on the same dominant phase --------------
    teleop_stack = dominant_phases.count("teleop")
    max_stack = max((dominant_phases.count(p) for p in distinct_phases), default=0)
    overlap_penalty = 0.0
    if teleop_stack == 3:
        overlap_penalty = 2.5   # all three fighting for the same cycles
    elif max_stack == 3:
        overlap_penalty = 2.0
    elif max_stack == 2:
        overlap_penalty = 1.0

    # --- Top-heaviness: EPA concentration in the single best robot --------------
    top_share = max(epas) / epa_sum
    # 0.33 (perfectly even) -> resilient ~10; 0.6+ (one robot carries) -> fragile
    top_heaviness = _clamp(10.0 - (top_share - 0.33) / 0.30 * 9.0)

    # --- Endgame strength: weighted endgame contribution across the alliance ----
    endgame_pts = sum(max(s.get("endgame", 0.0), 0.0) for s in stats_list)
    endgame_strength = _clamp(endgame_pts / 6.0)  # ~60 endgame pts -> 10

    # --- Overall chemistry blend ------------------------------------------------
    raw = (
        phase_coverage * 0.30 +
        top_heaviness * 0.30 +
        endgame_strength * 0.20 +
        (10.0 - overlap_penalty) * 0.20
    )
    overall = _clamp(raw)

    notes = []
    if len(distinct_phases) == 3:
        notes.append("Full phase coverage — auto, teleop, and endgame all represented.")
    if teleop_stack == 3:
        notes.append("All three teams are teleop-dominant — expect contested cycles / game-piece scarcity.")
    if top_share >= 0.55:
        notes.append(f"Top-heavy: one robot is {top_share*100:.0f}% of alliance EPA — fragile if it has an off match.")
    if endgame_pts < 10:
        notes.append("Thin endgame — limited climb/endgame ceiling.")
    if not notes:
        notes.append("Well-balanced alliance with complementary roles.")

    return {
        "overall_chemistry": overall,
        "phase_coverage": phase_coverage,
        "overlap_penalty": round(overlap_penalty, 2),
        "top_heaviness": top_heaviness,
        "top_share": round(top_share, 3),
        "endgame_strength": endgame_strength,
        "distinct_phases": sorted(distinct_phases),
        "notes": notes,
        "summary": (
            f"Alliance chemistry {overall:.1f}/10 — "
            f"coverage {phase_coverage:.0f}, balance {top_heaviness:.0f}, "
            f"endgame {endgame_strength:.0f}. " + " ".join(notes)
        ),
    }
