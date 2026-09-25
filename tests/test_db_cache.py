"""SQLite cache layer: TTL behaviour and key derivation."""

import time

import pytest


def test_roundtrip(temp_cache_db):
    db = temp_cache_db
    db.cache_set("k", {"a": 1}, ttl=60)
    assert db.cache_get("k") == {"a": 1}


def test_missing_key_is_none(temp_cache_db):
    assert temp_cache_db.cache_get("nope") is None


def test_expired_entry_reads_as_missing(temp_cache_db):
    """A TTL in the past must not be served — this is the whole point."""
    db = temp_cache_db
    db.cache_set("k", "v", ttl=-1)
    assert db.cache_get("k") is None


def test_entry_inside_its_ttl_is_served(temp_cache_db):
    db = temp_cache_db
    db.cache_set("k", "v", ttl=3600)
    assert db.cache_get("k") == "v"


def test_ttl_boundary_uses_wall_clock(temp_cache_db, monkeypatch):
    db = temp_cache_db
    db.cache_set("k", "v", ttl=10)
    assert db.cache_get("k") == "v"
    real = time.time
    monkeypatch.setattr(db.time, "time", lambda: real() + 11)
    assert db.cache_get("k") is None, "entry outlived its TTL"


def test_set_overwrites(temp_cache_db):
    db = temp_cache_db
    db.cache_set("k", "first", ttl=60)
    db.cache_set("k", "second", ttl=60)
    assert db.cache_get("k") == "second"


def test_clear_empties_the_cache(temp_cache_db):
    db = temp_cache_db
    db.cache_set("a", 1, ttl=60)
    db.cache_set("b", 2, ttl=60)
    assert db.cache_clear() == 2
    assert db.cache_get("a") is None


def test_purge_expired_only_removes_stale_rows(temp_cache_db):
    db = temp_cache_db
    db.cache_set("fresh", 1, ttl=3600)
    db.cache_set("stale", 2, ttl=-1)
    db.cache_purge_expired()
    assert db.cache_get("fresh") == 1
    assert db.cache_get("stale") is None


def test_unserializable_value_never_raises(temp_cache_db):
    """A cache write must never take down the request that triggered it.

    Note the actual behaviour: cache_set uses json.dumps(default=str), so a
    non-JSON value is COERCED to its str() rather than skipped, despite the
    docstring saying "silently skip un-serializable payloads". Caching a set
    therefore reads back as a string. Recorded here so the mismatch is visible;
    see HANDOFF.md.
    """
    db = temp_cache_db
    db.cache_set("k", {1, 2, 3}, ttl=60)  # a set is not JSON
    assert db.cache_get("k") == "{1, 2, 3}"


def test_stats_reports_entry_count(temp_cache_db):
    db = temp_cache_db
    db.cache_set("a", 1, ttl=60)
    db.cache_set("b", 2, ttl=60)
    assert db.cache_stats()["entries"] == 2


def test_documented_ttls(temp_cache_db):
    db = temp_cache_db
    assert db.TTL_TBA == 24 * 3600
    assert db.TTL_PREDICTION == 3600
    assert db.TTL_SYNTHESIS == 12 * 3600


def test_keys_are_deterministic_and_order_sensitive(temp_cache_db):
    db = temp_cache_db
    assert db.tba_team_key(254, 2026) == db.tba_team_key(254, 2026)
    assert db.tba_team_key(254, 2026) != db.tba_team_key(254, 2025)
    a = db.prediction_key([1, 2, 3], [4, 5, 6], 2026)
    assert a == db.prediction_key([1, 2, 3], [4, 5, 6], 2026)
    assert a != db.prediction_key([4, 5, 6], [1, 2, 3], 2026), "red/blue must not collide"


def test_synthesis_key_varies_with_model_and_notes(temp_cache_db):
    db = temp_cache_db
    base = db.synthesis_key("strategy", [1, 2, 3], [4, 5, 6], 2026, "", "qwen3:8b")
    assert base != db.synthesis_key("strategy", [1, 2, 3], [4, 5, 6], 2026, "n", "qwen3:8b")
    assert base != db.synthesis_key("strategy", [1, 2, 3], [4, 5, 6], 2026, "", "other")
    assert base != db.synthesis_key("captain", [1, 2, 3], [4, 5, 6], 2026, "", "qwen3:8b")
