"""Role gating on EVERY route, discovered from app.routes (not hand-listed)."""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute

from app.main import create_app

PUBLIC = {("POST", "/auth/login"), ("POST", "/auth/signup")}
ROLES = ("scout", "strategist", "admin")


def _role_gate(dependant):
    for d in dependant.dependencies:
        allowed = getattr(d.call, "allowed_roles", None)
        if allowed is not None:
            return allowed
        found = _role_gate(d)
        if found is not None:
            return found
    return None


def _flatten(routes):
    # Newer FastAPI wraps include_router() results; unwrap to the real APIRoutes.
    for r in routes:
        inner = getattr(r, "original_router", None)
        if inner is not None:
            yield from _flatten(inner.routes)
        else:
            yield r


def _routes(app):
    out = []
    for r in _flatten(app.routes):
        if isinstance(r, APIRoute):
            for m in sorted(r.methods - {"HEAD", "OPTIONS"}):
                out.append((m, r, r.path))
    return out


def _concrete(path: str) -> str:
    import re

    return re.sub(r"\{[^}]+\}", "1", path).replace("/event/1", "/event/2026test")


def test_no_unexpected_non_api_routes(app):
    # docs/openapi are disabled, so everything is an APIRoute
    from starlette.routing import Route

    for r in _flatten(app.routes):
        assert not (isinstance(r, Route) and not isinstance(r, APIRoute)), r.path


def test_every_route_has_role_gate_except_public(app):
    assert _routes(app), "no routes found"
    for method, route, path in _routes(app):
        gate = _role_gate(route.dependant)
        if (method, path) in PUBLIC:
            assert gate is None, f"{path} should be public"
        else:
            assert gate is not None, f"{method} {path} has no require_role dependency"


def test_anonymous_is_401_everywhere(app, client):
    for method, _route, path in _routes(app):
        if (method, path) in PUBLIC:
            continue
        r = client.request(method, _concrete(path), json={})
        assert r.status_code == 401, f"{method} {path} -> {r.status_code}"


@pytest.mark.parametrize("role", ROLES)
def test_role_matrix(app, make_user, login, role, mock_tba):
    import httpx

    mock_tba["handler"] = lambda req: httpx.Response(404)
    make_user(f"u_{role}", role)
    c = login(f"u_{role}")
    for method, route, path in _routes(app):
        if (method, path) in PUBLIC or path == "/auth/logout":  # logout kills the session
            continue
        allowed = _role_gate(route.dependant)
        permitted = not allowed or role == "admin" or role in allowed
        r = c.request(method, _concrete(path), json={})
        if permitted:
            assert r.status_code != 403, f"{role} blocked from {method} {path}"
            assert r.status_code != 401, f"{role} unauth on {method} {path}"
        else:
            assert r.status_code == 403, f"{role} reached {method} {path}: {r.status_code}"


def test_admin_routes_are_admin_only(app):
    admin_paths = [(m, p) for m, r, p in _routes(app) if p.startswith("/admin")]
    assert admin_paths
    for m, p in admin_paths:
        route = next(r for mm, r, pp in _routes(app) if (mm, pp) == (m, p))
        assert _role_gate(route.dependant) == frozenset({"admin"})


def test_forged_and_garbage_cookies_rejected(client, app):
    name = app.state.settings.cookie_name
    for bad in ("garbage", "a.b.c", ""):
        client.cookies.set(name, bad)
        assert client.get("/auth/me").status_code == 401


def test_token_signed_with_other_key_rejected(client, make_user, app):
    from app.security import create_token

    u = make_user("someone")
    client.cookies.set(app.state.settings.cookie_name, create_token(u.id, "w" * 48, 60))
    assert client.get("/auth/me").status_code == 401


def test_create_app_factory_smoke():
    assert create_app  # imported without side effects
