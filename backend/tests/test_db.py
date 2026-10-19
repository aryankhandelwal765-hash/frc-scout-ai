import sqlite3

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from app.db import make_engine
from app.models import Base

EXPECTED_TABLES = {
    "users",
    "invite_codes",
    "app_settings",
    "audit_log",
    "seasons",
    "events",
    "frc_teams",
    "matches",
    "match_teams",
    "external_stats",
    "scouting_entries",
    "pit_scouting",
    "pit_photos",
    "prescout_batches",
    "prescout_entries",
    "video_tracks",
    "ai_reports",
    "picklists",
    "picklist_items",
}


def test_pragmas_on_every_connection(settings):
    engine = make_engine(settings.database_url)
    conns = [engine.connect() for _ in range(3)]  # distinct pooled connections
    try:
        for c in conns:
            assert c.execute(text("PRAGMA foreign_keys")).scalar() == 1
            assert c.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    finally:
        for c in conns:
            c.close()


def test_app_sessions_have_pragmas(app, db):
    assert db.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_migrations_create_full_schema(db):
    names = set(inspect(db.get_bind()).get_table_names())
    assert EXPECTED_TABLES <= names


def test_migration_matches_models(db):
    insp = inspect(db.get_bind())
    for table in Base.metadata.sorted_tables:
        db_cols = {c["name"] for c in insp.get_columns(table.name)}
        assert db_cols == {c.name for c in table.columns}, table.name
        db_idx = {
            i["name"] for i in insp.get_indexes(table.name) if not i["name"].startswith("sqlite_")
        }
        model_idx = {i.name for i in table.indexes}
        assert model_idx <= db_idx, table.name


def test_invite_code_prefix_column_exists(db):
    assert "code_prefix" in {c["name"] for c in inspect(db.get_bind()).get_columns("invite_codes")}


def test_foreign_keys_enforced(db):
    db.execute(text("INSERT INTO frc_teams(number) VALUES (1)"))
    db.commit()
    with pytest.raises(IntegrityError):
        db.execute(
            text(
                "INSERT INTO match_teams(match_key, team_number, alliance) VALUES ('nope',1,'red')"
            )
        )
        db.commit()
    db.rollback()


def test_check_constraints(db):
    db.execute(text("INSERT INTO frc_teams(number) VALUES (1)"))
    db.execute(text("INSERT INTO seasons(year, scouting_schema) VALUES (2026,'{}')"))
    db.execute(text("INSERT INTO events(key, season) VALUES ('2026x',2026)"))
    db.execute(text("INSERT INTO prescout_batches(name, source_type) VALUES ('b','csv')"))
    db.commit()
    for sql in (
        "INSERT INTO prescout_batches(name, source_type) VALUES ('b','bogus')",
        "INSERT INTO prescout_entries(batch_id, team_number, data, confidence) VALUES (1,1,'{}',9)",
        "INSERT INTO matches(key,event_key) VALUES ('m','2026x'); "
        "INSERT INTO match_teams(match_key,team_number,alliance) VALUES ('m',1,'green')",
    ):
        with pytest.raises((IntegrityError, sqlite3.IntegrityError)):
            for stmt in sql.split(";"):
                db.execute(text(stmt))
            db.commit()
        db.rollback()


def test_downgrade_roundtrip(tmp_path):
    from alembic import command
    from alembic.config import Config

    from app.config import BACKEND_DIR

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{tmp_path / 'rt.db'}")
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
