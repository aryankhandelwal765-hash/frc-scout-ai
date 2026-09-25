"""win_probability_from_diff — the sigmoid behind every win probability."""

import math

import pytest

from scout.engine.feature_engineering import win_probability_from_diff


def test_zero_differential_is_a_coin_flip():
    assert win_probability_from_diff(0.0) == pytest.approx(0.5)


def test_positive_differential_favours_red():
    assert win_probability_from_diff(40.0) > 0.5


def test_negative_differential_favours_blue():
    assert win_probability_from_diff(-40.0) < 0.5


def test_symmetric_about_zero():
    """p(+d) and p(-d) must sum to exactly 1, or the two alliances disagree."""
    for d in (1.0, 17.5, 60.0, 250.0):
        assert win_probability_from_diff(d) + win_probability_from_diff(-d) == pytest.approx(1.0)


def test_monotonic_in_the_differential():
    diffs = [-200, -50, -10, 0, 10, 50, 200]
    probs = [win_probability_from_diff(d) for d in diffs]
    assert probs == sorted(probs)


def test_bounded_to_a_probability():
    for d in (-10_000.0, -300.0, 0.0, 300.0, 10_000.0):
        assert 0.0 <= win_probability_from_diff(d) <= 1.0


def test_larger_sigma_pulls_towards_even():
    """Sigma is the calibration knob. A wider sigma must be less confident."""
    tight = win_probability_from_diff(50.0, sigma=35.0)
    wide = win_probability_from_diff(50.0, sigma=250.0)
    assert 0.5 < wide < tight


def test_matches_the_logistic_definition():
    assert win_probability_from_diff(30.0, sigma=35.0) == pytest.approx(
        1.0 / (1.0 + math.exp(-30.0 / 35.0))
    )


def test_prediction_engine_uses_sigma_100():
    """Guard the documented miscalibration so a change is deliberate.

    The engine calls the sigmoid with sigma=100 while OPR is on a ~300/team
    scale, which is why win probabilities peg high. Recorded, not endorsed —
    see Known Limitations in the README.
    """
    import inspect

    from scout.engine import prediction_engine

    src = inspect.getsource(prediction_engine.predict_match)
    assert "sigma=100.0" in src
