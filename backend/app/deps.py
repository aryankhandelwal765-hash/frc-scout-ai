"""Auth dependencies. Every non-public route MUST use require_role()."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.security import decode_token


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    settings = request.app.state.settings
    token = request.cookies.get(settings.cookie_name)
    if not token:
        raise HTTPException(401, "Not authenticated")
    user_id = decode_token(token, settings.secret_key.get_secret_value())
    if user_id is None:
        raise HTTPException(401, "Not authenticated")
    # Re-check the DB on every request so deactivation is immediate.
    user = db.get(User, user_id)
    if user is None or not user.active:
        raise HTTPException(401, "Not authenticated")
    return user


def require_role(*roles: str) -> Callable[..., User]:
    """Dependency factory. Admins pass every role gate. Empty roles = any logged-in user."""
    allowed = frozenset(roles)

    def dep(user: User = Depends(current_user)) -> User:
        if allowed and user.role != "admin" and user.role not in allowed:
            raise HTTPException(403, "Insufficient role")
        return user

    dep.allowed_roles = allowed  # type: ignore[attr-defined]  # introspected by tests
    return dep


ANY_USER = require_role()
ADMIN = require_role("admin")
