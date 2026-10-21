"""history_ratings: probability from a ratings file, and graceful absence."""

import json

import pytest

from scout.engine import history_ratings as hr


@pytest.fixture
def ratings(tmp_path, monkeypatch):
    p = tmp_path / "team_ratings.json"
    p.write_text(json.dumps({"logistic_scale": 400, "ratings": {"1": 1900, "2": 1500, "3": 1100}}))
    monkeypatch.setenv("HISTORY_RATINGS", str(p))
    hr._cache.clear()
    return p


def test_stronger_alliance_is_favoured(ratings):
    r = hr.win_prob([1], [3])
    assert r["red_win_prob"] > 0.9 and r["unknown_teams"] == []
    assert hr.win_prob([3], [1])["red_win_prob"] < 0.1


def test_equal_alliances_are_even_and_probabilities_sum_to_one(ratings):
    r = hr.win_prob([2], [2])
    assert r["red_win_prob"] == pytest.approx(0.5)
    assert r["red_win_prob"] + r["blue_win_prob"] == pytest.approx(1.0)


def test_unknown_teams_count_as_average_and_are_reported(ratings):
    r = hr.win_prob([99], [2])
    assert r["red_win_prob"] == pytest.approx(0.5) and r["unknown_teams"] == [99]


def test_no_file_means_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HISTORY_RATINGS", str(tmp_path / "missing.json"))
    hr._cache.clear()
    assert hr.win_prob([1], [2]) is None and hr.rating(1) is None
