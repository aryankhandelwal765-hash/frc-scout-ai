"""tba_loader — phase-point extraction and the 2026 endgame rule. No network."""

from scout.engine import tba_loader as tba


def test_missing_breakdown_yields_zeros():
    assert tba._phase_points(None) == (0.0, 0.0, 0.0)
    assert tba._phase_points({}) == (0.0, 0.0, 0.0)
    assert tba._phase_points("not a dict") == (0.0, 0.0, 0.0)


def test_reads_the_2026_total_keys():
    auto, teleop, _endgame = tba._phase_points({"totalAutoPoints": 30, "totalTeleopPoints": 90})
    assert auto == 30.0
    assert teleop == 90.0


def test_reads_the_older_key_spelling():
    """TBA breakdown keys change every season; both spellings must work."""
    auto, teleop, _ = tba._phase_points({"autoPoints": 12, "teleopPoints": 40})
    assert (auto, teleop) == (12.0, 40.0)


def test_keys_are_matched_case_insensitively():
    auto, teleop, _ = tba._phase_points({"TOTALAUTOPOINTS": 5, "totalteleoppoints": 7})
    assert (auto, teleop) == (5.0, 7.0)


def test_endgame_is_hard_zeroed_for_2026():
    """Climb is worth ~nothing this season. Even when TBA reports points for it,
    the loader must return zero so no downstream engine weights it."""
    _auto, _teleop, endgame = tba._phase_points(
        {"totalAutoPoints": 20, "totalTeleopPoints": 80, "endGameTowerPoints": 45}
    )
    assert endgame == 0.0


def test_endgame_is_not_subtracted_out_of_teleop():
    """Those cycles still happened — they keep counting as teleop."""
    _auto, teleop, _endgame = tba._phase_points({"totalTeleopPoints": 80, "endGameTowerPoints": 45})
    assert teleop == 80.0


def test_negative_points_are_clamped():
    auto, teleop, _ = tba._phase_points({"autoPoints": -10, "teleopPoints": -5})
    assert auto == 0.0 and teleop == 0.0


def test_non_numeric_values_are_ignored():
    """TBA serialises some empty fields as the string "None"."""
    auto, teleop, _ = tba._phase_points({"autoPoints": "None", "teleopPoints": 40})
    assert auto == 0.0
    assert teleop == 40.0


def test_is_configured_follows_the_key(monkeypatch):
    monkeypatch.setattr(tba, "get_tba_key", lambda: None)
    assert tba.is_configured() is False
    monkeypatch.setattr(tba, "get_tba_key", lambda: "abc123")
    assert tba.is_configured() is True


def test_match_teams_strips_the_frc_prefix():
    match = {"alliances": {"red": {"team_keys": ["frc254", "frc1706", "frc2056"]}}}
    assert tba._match_teams(match, "red") == [254, 1706, 2056]


def test_team_color_identifies_the_side():
    match = {
        "alliances": {
            "red": {"team_keys": ["frc254"]},
            "blue": {"team_keys": ["frc1706"]},
        }
    }
    assert tba._team_color(match, 254) == "red"
    assert tba._team_color(match, 1706) == "blue"
    assert tba._team_color(match, 9999) is None
