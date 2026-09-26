"""data_source — the provider dispatcher. No provider is ever really called."""

import types

import pytest


@pytest.fixture
def fake_provider():
    """A stand-in module with the provider interface."""
    m = types.ModuleType("fake_provider")
    m.calls = []
    m.get_team_stats = lambda team, year: (
        m.calls.append(("stats", team, year)) or {"team": team, "epa": 42.0}
    )
    m.is_reachable = lambda: True
    m.is_configured = lambda: True
    m.get_event_matches = lambda key: m.calls.append(("matches", key)) or [{"key": key}]
    return m


@pytest.fixture
def ds(monkeypatch, fake_provider):
    from scout.engine import data_source

    monkeypatch.setitem(data_source._cache, "tba", fake_provider)
    monkeypatch.setitem(data_source._cache, "statbotics", fake_provider)
    return data_source


def test_get_team_stats_routes_to_the_active_provider(ds, fake_provider):
    assert ds.get_team_stats(254, 2026) == {"team": 254, "epa": 42.0}
    assert ("stats", 254, 2026) in fake_provider.calls


def test_get_event_matches_routes_through(ds, fake_provider):
    assert ds.get_event_matches("2026hop") == [{"key": "2026hop"}]
    assert ("matches", "2026hop") in fake_provider.calls


def test_unknown_provider_falls_back_to_tba(ds, monkeypatch, fake_provider):
    """A typo'd DATA_PROVIDER must not crash the app."""
    monkeypatch.setattr(ds, "DATA_PROVIDER", "not-a-provider")
    assert ds._provider() is fake_provider


def test_is_configured_defaults_true_when_provider_omits_it(ds, monkeypatch):
    """Providers without an API key need not implement is_configured()."""
    bare = types.ModuleType("bare")
    bare.is_reachable = lambda: True
    monkeypatch.setitem(ds._cache, "tba", bare)
    monkeypatch.setattr(ds, "DATA_PROVIDER", "tba")
    assert ds.is_configured() is True


def test_get_event_matches_empty_when_unsupported(ds, monkeypatch):
    """Statbotics has no schedule endpoint; the dispatcher returns []."""
    bare = types.ModuleType("bare")
    bare.is_reachable = lambda: True
    monkeypatch.setitem(ds._cache, "statbotics", bare)
    monkeypatch.setattr(ds, "DATA_PROVIDER", "statbotics")
    assert ds.get_event_matches("2026hop") == []


def test_providers_are_imported_lazily():
    """Only the active backend is imported, so statbotics need not be installed."""
    import inspect

    from scout.engine import data_source

    assert "importlib" in inspect.getsource(data_source)
    assert set(data_source._MODULES) == {"tba", "statbotics"}


def test_provider_name_reports_config(ds):
    assert ds.provider_name() == ds.DATA_PROVIDER
