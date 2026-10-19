from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.audit import audit
from app.db import get_db
from app.deps import ANY_USER
from app.models import InviteCode, User
from app.schemas import LoginIn, OkOut, SignupIn, UserOut
from app.security import burn_verify, create_token, hash_secret, verify_secret

router = APIRouter(prefix="/auth", tags=["auth"])

INVITE_PREFIX_LEN = 8


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _set_cookie(request: Request, response: Response, user: User) -> None:
    s = request.app.state.settings
    token = create_token(user.id, s.secret_key.get_secret_value(), s.token_ttl_seconds)
    response.set_cookie(
        s.cookie_name,
        token,
        max_age=s.token_ttl_seconds,
        httponly=True,
        secure=s.cookie_secure,
        samesite="strict",
        path="/",
    )


def _limited(request: Request, key: tuple[str, str]) -> None:
    wait = request.app.state.limiter.retry_after(key)
    if wait:
        raise HTTPException(429, "Too many attempts", headers={"Retry-After": str(wait)})


@router.post("/login", response_model=UserOut, responses={401: {}, 429: {}})
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    limiter = request.app.state.limiter
    username = body.username.strip().lower()
    key = (username, _client_ip(request))
    _limited(request, key)

    user = db.scalar(select(User).where(User.username == username))
    ok = False
    if user is None:
        burn_verify()
    else:
        ok = verify_secret(user.password_hash, body.password) and bool(user.active)
    if not ok:
        limiter.record_failure(key)
        audit(db, user.id if user else None, "login_failure", f"username={username[:64]}")
        db.commit()
        raise HTTPException(401, "Invalid username or password")

    limiter.reset(key)
    audit(db, user.id, "login_success")
    db.commit()
    _set_cookie(request, response, user)
    return user


@router.post("/logout", response_model=OkOut)
def logout(request: Request, response: Response, _user: User = Depends(ANY_USER)):
    s = request.app.state.settings
    response.delete_cookie(
        s.cookie_name, path="/", httponly=True, secure=s.cookie_secure, samesite="strict"
    )
    return OkOut()


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(ANY_USER)):
    return user


@router.post(
    "/signup", response_model=UserOut, status_code=201, responses={400: {}, 409: {}, 429: {}}
)
def signup(body: SignupIn, request: Request, response: Response, db: Session = Depends(get_db)):
    limiter = request.app.state.limiter
    key = ("signup", _client_ip(request))
    _limited(request, key)

    def bad_code():
        limiter.record_failure(key)
        audit(db, None, "signup_failure", "invalid invite code")
        db.commit()
        return HTTPException(400, "Invalid or expired invite code")

    code = body.invite_code.strip()
    invite = db.scalar(select(InviteCode).where(InviteCode.code_prefix == code[:INVITE_PREFIX_LEN]))
    if invite is None:
        burn_verify()
        raise bad_code()
    valid = verify_secret(invite.code_hash, code)
    if not valid or invite.used_by is not None:
        raise bad_code()
    if invite.expires_at and invite.expires_at <= datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"):
        raise bad_code()

    username = body.username.lower()
    user = User(
        username=username,
        display_name=body.display_name,
        password_hash=hash_secret(body.password),
        role=invite.role,
        active=1,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Username already taken") from None
    # Atomic single-use claim: only one concurrent signup can win.
    claimed = db.execute(
        update(InviteCode)
        .where(InviteCode.id == invite.id, InviteCode.used_by.is_(None))
        .values(used_by=user.id)
    )
    if claimed.rowcount != 1:
        db.rollback()
        raise bad_code()
    audit(
        db,
        user.id,
        "user_create",
        f"username={username} role={user.role} via=invite:{invite.id}",
    )
    db.commit()
    _set_cookie(request, response, user)
    return user
