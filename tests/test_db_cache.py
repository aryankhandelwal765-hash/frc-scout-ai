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


def test_unserializable_value_is_skipped_not_coerced(temp_cache_db):
    """A non-JSON value must be skipped, leaving a cache miss.

    It used to be coerced: json.dumps(default=str) turned a set into the string
    "{1, 2, 3}", so the cache handed back a different type than was written.
    A miss is recoverable; a silently wrong type is not.
    """
    db = temp_cache_db
    db.cache_set("k", {1, 2, 3}, ttl=60)  # a set is not JSON
    assert db.cache_get("k") is None


def test_unserializable_value_never_raises(temp_cache_db):
    """A cache write must never take down the request that triggered it."""
    class Opaque:
        pass

    db = temp_cache_db
    db.cache_set("k", Opaque(), ttl=60)      # no exception
    db.cache_set("nested", {"ok": 1, "bad": Opaque()}, ttl=60)
    assert db.cache_get("k") is None
    assert db.cache_get("nested") is None


def test_skipping_does_not_clobber_an_existing_good_entry(temp_cache_db):
    """A failed write must leave the previous value alone."""
    db = temp_cache_db
    db.cache_set("k", {"a": 1}, ttl=60)
    db.cache_set("k", {1, 2, 3}, ttl=60)     # skipped
    assert db.cache_get("k") == {"a": 1}


def test_skip_is_logged_at_debug(temp_cache_db, caplog):
    import logging

    db = temp_cache_db
    with caplog.at_level(logging.DEBUG, logger="scout.db"):
        db.cache_set("k", {1, 2, 3}, ttl=60)
    assert [r for r in caplog.records if r.levelno == logging.DEBUG
            and "not JSON-serializable" in r.getMessage()]


def test_real_payload_shapes_still_round_trip(temp_cache_db):
    """The shapes actual callers cache: TBA stats and vision results."""
    db = temp_cache_db
    stats = {"team": 254, "epa": 62.5, "auto": 15.0, "teleop": 47.5,
             "endgame": 0.0, "rank": None, "name": "The Cheesy Poofs"}
    matches = [{"key": "2026hop_qm1", "red": [1, 2, 3], "blue": [4, 5, 6]}]
    db.cache_set("stats", stats, ttl=60)
    db.cache_set("matches", matches, ttl=60)
    db.cache_set("empty", [], ttl=60)
    assert db.cache_get("stats") == stats
    assert db.cache_get("matches") == matches
    assert db.cache_get("empty") == []


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
