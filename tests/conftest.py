"""Shared fixtures. Nothing here touches the network."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ── Network guard ─────────────────────────────────────────────────────────
# TBA, Statbotics and Ollama are all HTTP. Rather than trusting every test to
# mock its own calls, make a real socket connection fail loudly.

@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    import socket

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "network access attempted in a test — mock the provider instead"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def temp_cache_db(tmp_path, monkeypatch):
    """A fresh SQLite cache per test, via the CACHE_DB_PATH env override."""
    db_path = tmp_path / "cache.db"
    monkeypatch.setenv("CACHE_DB_PATH", str(db_path))
    for mod in [m for m in sys.modules if m == "scout.db"]:
        del sys.modules[mod]
    import scout.db as db

    monkeypatch.setattr(db, "CACHE_DB", db.Path(str(db_path)))
    db.init_db()
    return db


@pytest.fixture
def team_stats():
    """A plausible EPA stat block, shaped like what the loaders return."""
    def _make(epa=60.0, auto=15.0, teleop=45.0, endgame=0.0, **extra):
        stats = {
            "team": extra.pop("team", 254),
            "name": extra.pop("name", "Test Team"),
            "epa": epa,
            "epa_sd": extra.pop("epa_sd", 8.0),
            "epa_max": extra.pop("epa_max", epa + 20),
            # The engines read these short keys, not the *_points spellings.
            "auto": auto,
            "teleop": teleop,
            "endgame": endgame,
            "wins": 10, "losses": 2, "ties": 0,
            "rank": 5, "percentile": 0.9, "team_count": 60,
        }
        stats.update(extra)
        return stats
    return _make


# ── Optional-asset markers ────────────────────────────────────────────────

def _yolo_weights_present() -> bool:
    for pattern in ("weights/*.pt", "runs/detect/*/weights/best.pt", "*.pt"):
        if list((ROOT / "vision").glob(pattern)):
            return True
    return False


needs_weights = pytest.mark.skipif(
    not _yolo_weights_present(),
    reason="no YOLO weights on disk (vision/weights/*.pt or vision/runs/**/best.pt)",
)

needs_opencv = pytest.mark.skipif(
    __import__("importlib").util.find_spec("cv2") is None,
    reason="opencv not installed",
)
