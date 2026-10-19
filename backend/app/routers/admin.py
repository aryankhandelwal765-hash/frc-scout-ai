from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.audit import audit
from app.db import get_db
from app.deps import ADMIN
from app.models import InviteCode, User
from app.routers.auth import INVITE_PREFIX_LEN
from app.schemas import ImportSummaryOut, InviteCreatedOut, InviteCreateIn, UserOut
from app.security import hash_secret
from app.services import tba

router = APIRouter(prefix="/admin", tags=["admin"])


def get_tba_client(request: Request):
    """Dependency so tests can swap in a mocked transport."""
    s = request.app.state.settings
    with httpx.Client(base_url=s.tba_base_url, timeout=15.0) as client:
        yield client


@router.post("/invites", response_model=InviteCreatedOut, status_code=201)
def create_invite(
    body: InviteCreateIn, db: Session = Depends(get_db), admin: User = Depends(ADMIN)
):
    code = secrets.token_urlsafe(24)
    expires = None
    if body.expires_in_hours:
        expires = (datetime.now(UTC) + timedelta(hours=body.expires_in_hours)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    invite = InviteCode(
        code_hash=hash_secret(code), code_prefix=code[:INVITE_PREFIX_LEN], role=body.role,
        expires_at=expires, created_by=admin.id,
    )
    db.add(invite)
    db.flush()
    audit(db, admin.id, "invite_create", f"invite={invite.id} role={body.role}")
    db.commit()
    return InviteCreatedOut(id=invite.id, code=code, role=body.role, expires_at=expires)


@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db), _admin: User = Depends(ADMIN)):
    return db.scalars(select(User).order_by(User.id)).all()


@router.post(
    "/users/{user_id}/deactivate", response_model=UserOut, responses={404: {}, 409: {}}
)
def deactivate_user(
    user_id: int = Path(ge=1), db: Session = Depends(get_db), admin: User = Depends(ADMIN)
):
    if user_id == admin.id:
        raise HTTPException(409, "You cannot deactivate yourself")
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(404, "User not found")
    if target.role == "admin" and target.active:
        n = db.scalar(
            select(func.count()).select_from(User).where(User.role == "admin", User.active == 1)
        )
        if n <= 1:
            raise HTTPException(409, "Cannot deactivate the last active admin")
    if target.active:
        target.active = 0
        audit(db, admin.id, "user_deactivate", f"target={target.id} username={target.username}")
        db.commit()
    return target


@router.post(
    "/import/event/{event_key}", response_model=ImportSummaryOut,
    responses={404: {}, 409: {}, 422: {}, 502: {}, 503: {}},
)
def import_event(
    request: Request,
    event_key: str = Path(pattern=tba.EVENT_KEY_RE.pattern),
    db: Session = Depends(get_db),
    admin: User = Depends(ADMIN),
    client: httpx.Client = Depends(get_tba_client),
):
    key = request.app.state.settings.tba_api_key.get_secret_value()
    try:
        summary = tba.import_event(db, client, key, event_key)
    except tba.TbaError as e:
        db.rollback()
        raise HTTPException(e.status, str(e)) from None
    audit(db, admin.id, "event_import", f"event={event_key}")
    db.commit()
    return summary
