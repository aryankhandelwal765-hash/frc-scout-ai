from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import BACKEND_DIR, Settings
from app.db import make_engine, make_session_factory
from app.main import create_app
from app.models import User
from app.routers.admin import get_tba_client
from app.security import hash_secret

PASSWORD = "correct-horse-battery"  # test-only value


@pytest.fixture(scope="session")
def template_db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tpl") / "template.db"
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    command.upgrade(cfg, "head")
    # Fold the WAL into the main file so a plain file copy is a complete database.
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    return path


@pytest.fixture
def settings(tmp_path, template_db) -> Settings:
    db = tmp_path / "test.db"
    shutil.copy(template_db, db)
    return Settings(
        _env_file=None,
        env="dev",
        secret_key="t" * 48,
        cookie_secure=True,
        database_url=f"sqlite:///{db}",
        tba_api_key="fake-key",
        cors_origins="",
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def db(app):
    with app.state.session_factory() as s:
        yield s


@pytest.fixture
def client(app):
    # https base so Secure cookies are sent back by the test client
    with TestClient(app, base_url="https://testserver") as c:
        yield c


@pytest.fixture
def make_user(app):
    def _make(username: str, role: str = "scout", active: int = 1) -> User:
        with app.state.session_factory() as s:
            u = User(
                username=username, password_hash=hash_secret(PASSWORD), role=role, active=active
            )
            s.add(u)
            s.commit()
            return u

    return _make


def new_client(app) -> TestClient:
    return TestClient(app, base_url="https://testserver")


@pytest.fixture
def login(app):
    def _login(username: str) -> TestClient:
        c = new_client(app)
        r = c.post("/auth/login", json={"username": username, "password": PASSWORD})
        assert r.status_code == 200, r.text
        return c

    return _login


@pytest.fixture
def admin_client(make_user, login):
    make_user("boss", "admin")
    return login("boss")


@pytest.fixture
def mock_tba(app):
    """Install a mocked TBA transport. Returns a state dict {handler, calls}."""
    state = {"calls": [], "handler": None}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"].append(request)
        return state["handler"](request)

    def _dep():
        with httpx.Client(base_url="https://tba.test", transport=httpx.MockTransport(handler)) as c:
            yield c

    app.dependency_overrides[get_tba_client] = _dep
    return state


@pytest.fixture
def sessionmaker_for(settings):
    return make_session_factory(make_engine(settings.database_url))


def user_by_name(db, name):
    return db.scalar(select(User).where(User.username == name))
