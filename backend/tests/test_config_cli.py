import pytest
from pydantic import ValidationError

from app.config import Settings


def mk(**kw):
    return Settings(_env_file=None, **kw)


def test_prod_requires_strong_secret():
    for bad in ("", "short", "changeme" * 6, "a" * 40, "my-secret-key-" + "q7Zx" * 8):
        with pytest.raises(ValidationError):
            mk(env="prod", secret_key=bad)
    with pytest.raises(ValidationError):
        mk(env="prod")  # missing entirely
    ok = mk(env="prod", secret_key="Xk9Qm2Lp7Vw4Rt8Yb3Nc6Hd1Fs5Jg0Za-Ue")
    assert ok.cookie_secure is True


def test_cookie_secure_false_only_in_dev():
    assert mk(env="dev", cookie_secure=False).cookie_secure is False
    with pytest.raises(ValidationError):
        mk(env="prod", cookie_secure=False, secret_key="Xk9Qm2Lp7Vw4Rt8Yb3Nc6Hd1Fs5Jg0Za-Ue")


def test_cookie_secure_defaults_true():
    assert mk(env="dev").cookie_secure is True


def test_dev_secret_is_random_not_constant():
    assert (
        mk(env="dev").secret_key.get_secret_value() != mk(env="dev").secret_key.get_secret_value()
    )


def test_env_alias_names(monkeypatch):
    for k in ("TBA_API_KEY", "TBA_AUTH_KEY", "OLLAMA_URL", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TBA_AUTH_KEY", "legacy")
    monkeypatch.setenv("OLLAMA_HOST", "http://old:1")
    s = mk()
    assert s.tba_api_key.get_secret_value() == "legacy" and s.ollama_url == "http://old:1"
    monkeypatch.setenv("TBA_API_KEY", "new")
    monkeypatch.setenv("OLLAMA_URL", "http://new:2")
    s = mk()
    assert s.tba_api_key.get_secret_value() == "new" and s.ollama_url == "http://new:2"


def test_cors_wildcard_rejected():
    with pytest.raises(ValidationError):
        mk(cors_origins="*")


def test_create_admin_cli(monkeypatch, settings, db):
    from app import cli
    from app.models import AuditLog, User

    monkeypatch.setenv("DATABASE_URL", settings.database_url)
    monkeypatch.setenv("SECRET_KEY", "t" * 48)
    monkeypatch.setenv("ADMIN_PASSWORD", "a-scripted-password")
    assert cli.main(["create-admin", "--username", "Root1"]) == 0
    assert cli.main(["create-admin", "--username", "Root1"]) == 1  # exists
    db.expire_all()
    u = db.query(User).filter_by(username="root1").one()
    assert u.role == "admin" and u.password_hash.startswith("$argon2")
    assert db.query(AuditLog).filter_by(action="user_create").count() == 1
    monkeypatch.setenv("ADMIN_PASSWORD", "short")
    assert cli.main(["create-admin", "--username", "other1"]) == 2


def test_seed_season_cli_idempotent(monkeypatch, settings, db):
    from app import cli
    from app.models import Season

    monkeypatch.setenv("DATABASE_URL", settings.database_url)
    assert cli.main(["seed-season"]) == 0
    assert cli.main(["seed-season"]) == 0
    assert db.query(Season).count() == 1
    assert "EXAMPLE" in db.get(Season, 2026).game_name
