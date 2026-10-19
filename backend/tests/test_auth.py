from __future__ import annotations

import pytest
from sqlalchemy import select

from app.models import AuditLog, InviteCode, User
from tests.conftest import PASSWORD, new_client


def actions(db):
    return [a.action for a in db.scalars(select(AuditLog).order_by(AuditLog.id))]


def test_login_sets_secure_cookie_flags(client, make_user):
    make_user("alice")
    r = client.post("/auth/login", json={"username": "alice", "password": PASSWORD})
    assert r.status_code == 200
    assert r.json()["username"] == "alice" and "password_hash" not in r.text
    sc = r.headers["set-cookie"].lower()
    assert "httponly" in sc and "secure" in sc and "samesite=strict" in sc
    assert client.get("/auth/me").json()["username"] == "alice"


def test_login_username_case_insensitive_and_bad_password(client, make_user, db):
    make_user("bob")
    assert (
        client.post("/auth/login", json={"username": "BOB", "password": "nope"}).status_code == 401
    )
    assert (
        client.post("/auth/login", json={"username": "ghost", "password": "x"}).status_code == 401
    )
    assert (
        client.post("/auth/login", json={"username": "BOB", "password": PASSWORD}).status_code
        == 200
    )
    assert actions(db) == ["login_failure", "login_failure", "login_success"]


def test_logout_clears_cookie(login, make_user):
    make_user("carl")
    c = login("carl")
    assert c.post("/auth/logout").status_code == 200
    assert c.get("/auth/me").status_code == 401


def test_deactivation_kills_old_cookie_immediately(app, make_user, login, admin_client):
    victim = make_user("victim", "scout")
    c = login("victim")
    assert c.get("/auth/me").status_code == 200
    r = admin_client.post(f"/admin/users/{victim.id}/deactivate")
    assert r.status_code == 200 and r.json()["active"] is False
    assert c.get("/auth/me").status_code == 401  # same cookie, next request
    r = new_client(app).post("/auth/login", json={"username": "victim", "password": PASSWORD})
    assert r.status_code == 401


def test_cannot_deactivate_self_or_last_admin(admin_client, make_user, db):
    me = admin_client.get("/auth/me").json()
    assert admin_client.post(f"/admin/users/{me['id']}/deactivate").status_code == 409
    # second admin can't remove the only *other* active admin if it makes zero... set up:
    other = make_user("admin2", "admin")
    assert admin_client.post(f"/admin/users/{other.id}/deactivate").status_code == 200
    assert admin_client.post("/admin/users/9999/deactivate").status_code == 404


def test_last_admin_guard_from_second_admin(app, make_user, login):
    a = make_user("a1", "admin")
    b = make_user("a2", "admin")
    ca, cb = login("a1"), login("a2")
    assert cb.post(f"/admin/users/{a.id}/deactivate").status_code == 200
    # a1 is now inactive, so b is the last admin; b is also self -> 409 either way
    assert cb.post(f"/admin/users/{b.id}/deactivate").status_code == 409
    assert ca.get("/auth/me").status_code == 401


def test_login_rate_limit(client, make_user):
    make_user("dave")
    for _ in range(5):
        assert (
            client.post("/auth/login", json={"username": "dave", "password": "bad"}).status_code
            == 401
        )
    r = client.post("/auth/login", json={"username": "dave", "password": PASSWORD})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    # other username from same IP is independent
    make_user("erin")
    assert (
        client.post("/auth/login", json={"username": "erin", "password": PASSWORD}).status_code
        == 200
    )


def test_success_resets_rate_limit(client, make_user):
    make_user("fay")
    for _ in range(4):
        client.post("/auth/login", json={"username": "fay", "password": "bad"})
    assert (
        client.post("/auth/login", json={"username": "fay", "password": PASSWORD}).status_code
        == 200
    )
    for _ in range(4):
        assert (
            client.post("/auth/login", json={"username": "fay", "password": "bad"}).status_code
            == 401
        )


def _invite(admin_client, **body):
    r = admin_client.post("/admin/invites", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _signup(c, code, username="newbie", password="a-long-enough-pw"):
    return c.post(
        "/auth/signup", json={"invite_code": code, "username": username, "password": password}
    )


def test_invite_flow_single_use(app, admin_client, db):
    inv = _invite(admin_client, role="strategist")
    stored = db.scalar(select(InviteCode).where(InviteCode.id == inv["id"]))
    assert stored.code_hash.startswith("$argon2") and inv["code"] not in stored.code_hash
    assert stored.code_prefix == inv["code"][:8]

    c = new_client(app)
    r = _signup(c, inv["code"])
    assert r.status_code == 201 and r.json()["role"] == "strategist"
    assert c.get("/auth/me").json()["username"] == "newbie"  # auto-login

    r2 = _signup(new_client(app), inv["code"], username="second")
    assert r2.status_code == 400  # already used
    db.expire_all()
    assert db.get(InviteCode, inv["id"]).used_by is not None


def test_invite_bad_and_expired_codes(app, admin_client, db):
    assert _signup(new_client(app), "totally-bogus-code").status_code == 400
    inv = _invite(admin_client, expires_in_hours=1)
    # prefix matches but secret wrong
    assert _signup(new_client(app), inv["code"][:8] + "x" * 20).status_code == 400
    row = db.get(InviteCode, inv["id"])
    row.expires_at = "2000-01-01 00:00:00"
    db.commit()
    assert _signup(new_client(app), inv["code"]).status_code == 400


def test_signup_duplicate_username_keeps_invite_unused(app, admin_client, make_user, db):
    make_user("taken")
    inv = _invite(admin_client)
    assert _signup(new_client(app), inv["code"], username="taken").status_code == 409
    assert _signup(new_client(app), inv["code"], username="free").status_code == 201


def test_signup_rejects_weak_input(app, admin_client):
    inv = _invite(admin_client)
    c = new_client(app)
    assert _signup(c, inv["code"], password="short").status_code == 422
    assert _signup(c, inv["code"], username="bad name!").status_code == 422


def test_signup_is_rate_limited(app):
    c = new_client(app)
    for _ in range(5):
        assert _signup(c, "bogus-bogus-bogus").status_code == 400
    assert _signup(c, "bogus-bogus-bogus").status_code == 429


def test_invite_requires_admin(make_user, login):
    make_user("s", "scout")
    make_user("st", "strategist")
    assert login("s").post("/admin/invites", json={}).status_code == 403
    assert login("st").post("/admin/invites", json={}).status_code == 403


def test_audit_rows(app, admin_client, make_user, db):
    inv = _invite(admin_client)
    _signup(new_client(app), inv["code"], username="zed")
    v = make_user("vic")
    admin_client.post(f"/admin/users/{v.id}/deactivate")
    new_client(app).post("/auth/login", json={"username": "boss", "password": "wrong"})
    db.expire_all()
    got = actions(db)
    for a in ("login_success", "invite_create", "user_create", "user_deactivate", "login_failure"):
        assert a in got, got
    rows = db.scalars(select(AuditLog)).all()
    assert all(
        PASSWORD not in (r.detail or "") and inv["code"] not in (r.detail or "") for r in rows
    )


def test_list_users_hides_hashes(admin_client, make_user):
    make_user("x1")
    r = admin_client.get("/admin/users")
    assert r.status_code == 200 and len(r.json()) >= 2
    assert "hash" not in r.text


@pytest.mark.parametrize("n", [1])
def test_users_table_role_check(db, n):
    from sqlalchemy.exc import IntegrityError

    db.add(User(username="bad", password_hash="x", role="root"))
    with pytest.raises(IntegrityError):
        db.commit()
