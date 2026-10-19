"""Password hashing, JWT, and in-process login rate limiting."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_ph = PasswordHasher()
# Hash of a random throwaway value, used to equalise timing for unknown usernames.
_DUMMY_HASH = _ph.hash("not-a-real-password-" + str(time.time_ns()))
JWT_ALG = "HS256"


def hash_secret(value: str) -> str:
    return _ph.hash(value)


def verify_secret(hashed: str, value: str) -> bool:
    try:
        return _ph.verify(hashed, value)
    except (VerificationError, InvalidHashError):
        return False


def burn_verify() -> None:
    verify_secret(_DUMMY_HASH, "x")


def create_token(user_id: int, secret: str, ttl_seconds: int) -> str:
    now = datetime.now(UTC)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(seconds=ttl_seconds)}
    return jwt.encode(payload, secret, algorithm=JWT_ALG)


def decode_token(token: str, secret: str) -> int | None:
    try:
        data = jwt.decode(token, secret, algorithms=[JWT_ALG], options={"require": ["exp", "sub"]})
        return int(data["sub"])
    except (jwt.PyJWTError, ValueError):
        return None


class RateLimiter:
    """Sliding-window failure counter. In-process only (single worker deployment)."""

    def __init__(self, max_failures: int, window_seconds: int) -> None:
        self.max = max_failures
        self.window = window_seconds
        self._fails: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, q: deque[float], now: float) -> None:
        while q and now - q[0] > self.window:
            q.popleft()

    def retry_after(self, key: tuple[str, str]) -> int:
        """Seconds until allowed again; 0 if not limited."""
        now = time.monotonic()
        with self._lock:
            q = self._fails[key]
            self._prune(q, now)
            if len(q) >= self.max:
                return max(1, int(self.window - (now - q[0])))
            return 0

    def record_failure(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._fails[key].append(time.monotonic())

    def reset(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._fails.pop(key, None)
