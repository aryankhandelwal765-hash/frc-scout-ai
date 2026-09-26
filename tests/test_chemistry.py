"""analyze_chemistry — alliance synergy from EPA-derived archetypes."""

import pytest

from scout.engine.chemistry_engine import analyze_chemistry


def test_empty_alliance_is_handled(team_stats):
    result = analyze_chemistry([])
    assert result["overall_chemistry"] == 0.0
    assert result["notes"]


def test_scores_stay_on_the_documented_1_to_10_scale(team_stats):
    result = analyze_chemistry([team_stats(), team_stats(epa=40.0), team_stats(epa=20.0)])
    for key in ("overall_chemistry", "phase_coverage", "top_heaviness", "endgame_strength"):
        assert 1.0 <= result[key] <= 10.0, f"{key} off scale: {result[key]}"


def test_even_alliance_beats_top_heavy_one(team_stats):
    """Top-heaviness is the fragility axis: one robot carrying scores worse."""
    even = analyze_chemistry([team_stats(epa=50.0) for _ in range(3)])
    carried = analyze_chemistry([team_stats(epa=150.0), team_stats(epa=10.0), team_stats(epa=10.0)])
    assert even["top_heaviness"] > carried["top_heaviness"]


def test_top_share_is_a_fraction(team_stats):
    result = analyze_chemistry([team_stats(epa=150.0), team_stats(epa=25.0), team_stats(epa=25.0)])
    assert result["top_share"] == pytest.approx(150 / 200, abs=1e-3)


def test_three_teleop_robots_are_flagged_as_contested(team_stats):
    teleop_bot = lambda: team_stats(auto=5.0, teleop=90.0, endgame=0.0)
    result = analyze_chemistry([teleop_bot(), teleop_bot(), teleop_bot()])
    assert result["overlap_penalty"] > 0
    assert any("contested" in n.lower() for n in result["notes"])


def test_diverse_phases_score_better_coverage_than_identical_ones(team_stats):
    same = [team_stats(auto=5.0, teleop=90.0, endgame=0.0) for _ in range(3)]
    mixed = [
        team_stats(auto=70.0, teleop=20.0, endgame=0.0),
        team_stats(auto=5.0, teleop=90.0, endgame=0.0),
        team_stats(auto=10.0, teleop=20.0, endgame=70.0),
    ]
    assert analyze_chemistry(mixed)["phase_coverage"] > analyze_chemistry(same)["phase_coverage"]


def test_zero_epa_alliance_does_not_divide_by_zero(team_stats):
    """Teams with no TBA data arrive as all-zero stats."""
    result = analyze_chemistry([team_stats(epa=0.0, auto=0, teleop=0, endgame=0)] * 3)
    assert 0.0 <= result["overall_chemistry"] <= 10.0


def test_notes_are_always_present(team_stats):
    result = analyze_chemistry([team_stats(), team_stats(), team_stats()])
    assert result["notes"] and all(isinstance(n, str) for n in result["notes"])
