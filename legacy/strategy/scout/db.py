"""SQLite persistence for FRC Scout AI.

Caches:
  - TBA team stats (24h TTL) — big win, TBA is rate-limited and slow.
  - TBA event matches (24h TTL) — same.
  - Prediction results (1h TTL) — quant re-computation is cheap but not free.
  - Synthesis reads (LLM outputs) (12h TTL) — 5-10s per call, worth caching.

Single-file SQLite database at CACHE_DB (env override CACHE_DB_PATH). Safe for
FastAPI's async workers — sqlite3 is thread-safe with `check_same_thread=False`
and short-lived connections. All reads/writes go through the helpers below.
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Database path (env override for containers/tests)
CACHE_DB = Path(os.environ.get("CACHE_DB_PATH", Path(__file__).resolve().parent / "cache.db"))

# TTLs (seconds)
TTL_TBA = 24 * 3600  # TBA team stats + event matches: 24h
TTL_PREDICTION = 3600  # quant predictions: 1h
TTL_SYNTHESIS = 12 * 3600  # LLM synthesis output: 12h


def _conn() -> sqlite3.Connection:
    """Short-lived connection; caller closes."""
    c = sqlite3.connect(str(CACHE_DB), check_same_thread=False, timeout=5.0)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """Create tables if missing. Idempotent; safe to call on every startup."""
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS kv_cache (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                created_at  REAL NOT NULL,
                ttl         REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kv_cache_expiry
                ON kv_cache (created_at, ttl);
        """)


def _hash(*parts: Any) -> str:
    """Deterministic hash of the arguments for a cache key."""
    payload = "|".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_get(key: str) -> Any | None:
    """Return the cached value or None if missing/expired."""
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT value, created_at, ttl FROM kv_cache WHERE key = ?", (key,)
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    if time.time() - row["created_at"] > row["ttl"]:
        return None  # expired (garbage-collected lazily on next set)
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return None


def cache_set(key: str, value: Any, ttl: float) -> None:
    """Persist a JSON-serializable value under key with a TTL (seconds).

    A value that is not JSON-serializable is skipped, not coerced. This used to
    pass default=str, which meant nothing was ever un-serializable: a set was
    stored as the string "{1, 2, 3}" and read back as that string, so the cache
    silently returned a different type than was written. Skipping keeps the
    cache honest — a miss is recoverable, a wrong type is not.
    """
    try:
        payload = json.dumps(value)
    except (TypeError, ValueError) as exc:
        logger.debug("cache_set(%s): value is not JSON-serializable, skipping (%s)", key, exc)
        return
    try:
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO kv_cache (key, value, created_at, ttl) VALUES (?, ?, ?, ?)",
                (key, payload, time.time(), ttl),
            )
    except sqlite3.Error:
        pass  # cache write failure must never break the request


def cache_clear() -> int:
    """Wipe all cached entries. Returns row count deleted."""
    with _conn() as c:
        cur = c.execute("DELETE FROM kv_cache")
        return cur.rowcount


def cache_purge_expired() -> int:
    """Delete expired rows. Run periodically to keep the DB small."""
    now = time.time()
    with _conn() as c:
        cur = c.execute("DELETE FROM kv_cache WHERE (? - created_at) > ttl", (now,))
        return cur.rowcount


def cache_stats() -> dict:
    """Diagnostic: count of cached entries, size on disk, oldest entry age."""
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n, MIN(created_at) AS oldest FROM kv_cache"
            ).fetchone()
        size_bytes = CACHE_DB.stat().st_size if CACHE_DB.exists() else 0
        oldest_age = (time.time() - row["oldest"]) if row["oldest"] else 0
        return {
            "entries": row["n"],
            "size_mb": round(size_bytes / 1e6, 2),
            "oldest_age_sec": int(oldest_age),
            "db_path": str(CACHE_DB),
        }
    except sqlite3.Error as e:
        return {"error": str(e), "db_path": str(CACHE_DB)}


# ── Domain-specific cache keys ─────────────────────────────────────────────────


def tba_team_key(team: int, year: int) -> str:
    return f"tba:team:{team}:{year}"


def tba_matches_key(event_key: str) -> str:
    return f"tba:matches:{event_key}"


def prediction_key(red: list[int], blue: list[int], year: int) -> str:
    # Sort each alliance so team order doesn't create separate cache entries.
    r = sorted(red)
    b = sorted(blue)
    return f"predict:{_hash(r, b, year)}"


def synthesis_key(
    kind: str, red: list[int], blue: list[int], year: int, notes: str, model: str
) -> str:
    """kind = 'strategy' | 'captain' | 'coach' | 'bots'. Notes are content-addressed."""
    r = sorted(red)
    b = sorted(blue)
    notes_h = hashlib.sha256((notes or "").encode()).hexdigest()[:8]
    return f"synth:{kind}:{_hash(r, b, year, notes_h, model)}"
