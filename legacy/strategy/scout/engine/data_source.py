"""Data-source dispatcher.

Routes team-stats lookups to the provider named in config.DATA_PROVIDER, so the
rest of the app (prediction_engine, pages) never imports a specific backend. Flip
config.DATA_PROVIDER between "tba" and "statbotics" to switch sources — e.g. back
to Statbotics once their API (down since 2026-07-09) recovers.

Providers are imported LAZILY — only the active one is loaded — so the app does
not require, say, the `statbotics` package installed just to run on TBA.

Every provider exposes the same interface: get_team_stats(team, year),
is_reachable(). TBA also exposes is_configured() (needs an API key); providers
without a key default is_configured() to True.
"""

import importlib

from scout.config import DATA_PROVIDER

_MODULES = {
    "tba": "scout.engine.tba_loader",
    "statbotics": "scout.engine.statbotics_loader",
}
_cache: dict[str, object] = {}


def _provider():
    name = DATA_PROVIDER if DATA_PROVIDER in _MODULES else "tba"
    if name not in _cache:
        _cache[name] = importlib.import_module(_MODULES[name])
    return _cache[name]


def provider_name() -> str:
    return DATA_PROVIDER


def is_configured() -> bool:
    """Whether the active provider has what it needs (e.g. an API key)."""
    p = _provider()
    return p.is_configured() if hasattr(p, "is_configured") else True


def is_reachable() -> bool:
    return _provider().is_reachable()


def get_team_stats(team: int, year: int = 2026) -> dict | None:
    return _provider().get_team_stats(team, year)


def get_event_matches(event_key: str) -> list[dict]:
    """Match schedule for an event (TBA feature). Empty if unsupported."""
    p = _provider()
    if hasattr(p, "get_event_matches"):
        return p.get_event_matches(event_key)
    return []
