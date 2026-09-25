"""FastAPI surface: routes exist and validation holds. No LLM, no network."""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(temp_cache_db, monkeypatch):
    from scout.api.main import app

    return TestClient(app)


def test_health_is_reachable(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "synthesis_model" in body and "bot_model" in body


def test_predict_requires_exactly_three_teams_per_alliance(client):
    r = client.post("/predict", json={"red_teams": [1, 2], "blue_teams": [4, 5, 6]})
    assert r.status_code == 422


def test_predict_rejects_four_teams(client):
    r = client.post("/predict", json={"red_teams": [1, 2, 3, 4], "blue_teams": [4, 5, 6]})
    assert r.status_code == 422


def test_render_status_for_an_unknown_match_is_idle(client):
    r = client.get("/scouting/render/definitely-not-a-match")
    assert r.status_code == 200
    assert r.json()["status"] == "idle"


def test_render_post_404s_without_preprocessing(client):
    r = client.post("/scouting/render/definitely-not-a-match")
    assert r.status_code == 404


def test_cache_stats_shape(client):
    r = client.get("/cache/stats")
    assert r.status_code == 200
    assert "entries" in r.json()


def test_documented_routes_are_all_registered(client):
    from scout.api.main import app

    paths = {getattr(r, "path", None) for r in app.routes}
    for path in (
        "/health", "/cache/stats", "/settings", "/predict", "/roles",
        "/analyze/stream", "/coach/stream", "/captain/stream", "/bots/stream",
        "/scouting/matches", "/scouting/render/{match_key}",
        "/vision/team/{team}", "/vision/match/{match_key}",
    ):
        assert path in paths, f"missing route {path}"


def test_render_endpoint_has_both_verbs():
    """POST queues the render, GET polls it — the fix for the blocking render."""
    from scout.api.main import app

    methods = set()
    for r in app.routes:
        if getattr(r, "path", None) == "/scouting/render/{match_key}":
            methods |= set(getattr(r, "methods", set()))
    assert {"POST", "GET"} <= methods
