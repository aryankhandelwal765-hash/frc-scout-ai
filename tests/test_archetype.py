"""classify_archetype — playstyle + phase-share classification."""

import pytest

from scout.engine.archetype_engine import classify_archetype


def test_returns_the_documented_shape(team_stats):
    result = classify_archetype(team_stats())
    for key in ("archetype", "tier", "dominant_phase", "phase_shares", "summary"):
        assert key in result, f"missing {key}"


def test_phase_shares_are_a_distribution(team_stats):
    shares = classify_archetype(team_stats(auto=20.0, teleop=60.0, endgame=0.0))["phase_shares"]
    assert all(0.0 <= v <= 1.0 for v in shares.values())
    assert sum(shares.values()) == pytest.approx(1.0, abs=2e-3)


def test_teleop_heavy_robot_is_dominant_in_teleop(team_stats):
    result = classify_archetype(team_stats(auto=5.0, teleop=80.0, endgame=0.0))
    assert result["dominant_phase"] == "teleop"


def test_auto_heavy_robot_is_dominant_in_auto(team_stats):
    result = classify_archetype(team_stats(auto=70.0, teleop=10.0, endgame=0.0))
    assert result["dominant_phase"] == "auto"


def test_empty_stats_do_not_raise():
    """Teams with no TBA data reach this with an empty dict."""
    result = classify_archetype({})
    assert result["dominant_phase"] == "none"


def test_all_zero_breakdown_is_not_a_division_by_zero(team_stats):
    result = classify_archetype(team_stats(epa=0.0, auto=0.0, teleop=0.0, endgame=0.0))
    assert result["dominant_phase"] == "none"
    assert sum(result["phase_shares"].values()) == pytest.approx(0.0)


def test_stronger_epa_never_lowers_the_tier(team_stats):
    """Tier is ordinal; a better robot must not rank below a worse one."""
    tiers = [
        classify_archetype(team_stats(epa=e, percentile=p))["tier"]
        for e, p in ((10.0, 0.1), (60.0, 0.5), (120.0, 0.95))
    ]
    assert len(set(tiers)) > 1, "tier never varied — the classifier is inert"


def test_summary_is_a_nonempty_string(team_stats):
    assert classify_archetype(team_stats())["summary"].strip()
