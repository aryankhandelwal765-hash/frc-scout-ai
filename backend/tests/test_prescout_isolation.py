"""Pre-scout code paths must only write prescout_* tables, never scouting_entries."""

import inspect as pyinspect
import re

import pytest
from sqlalchemy import event, text

from app.models import FrcTeam, ScoutingEntry, User
from app.services import prescout, seasons
from app.services.schema_validate import PayloadInvalid

WRITE = re.compile(r"^\s*(INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE|DELETE\s+FROM)\s+[\"`]?(\w+)", re.I)


@pytest.fixture
def seeded(db):
    seasons.seed_example_season(db)
    db.add(FrcTeam(number=254, name="x"))
    db.add(User(username="p", password_hash="x", role="scout"))
    db.commit()
    return db


def test_prescout_writes_only_touch_prescout_tables(seeded):
    db = seeded
    written: set[str] = set()

    @event.listens_for(db.get_bind(), "before_cursor_execute")
    def spy(conn, cursor, statement, params, context, executemany):
        m = WRITE.match(statement)
        if m:
            written.add(m.group(2).lower())

    uid = db.query(User).one().id
    batch = prescout.create_batch(db, name="b", source_type="manual", created_by=uid)
    prescout.add_entry(
        db,
        batch_id=batch.id,
        team_number=254,
        season_year=2026,
        data={"drivetrain": "swerve"},
        scouter_id=uid,
        confidence=4,
    )
    db.commit()

    assert written == {"prescout_batches", "prescout_entries"}
    assert db.execute(text("SELECT COUNT(*) FROM scouting_entries")).scalar() == 0
    assert db.execute(text("SELECT COUNT(*) FROM prescout_entries")).scalar() == 1


def test_prescout_entry_validated_against_prescout_schema(seeded):
    db = seeded
    batch = prescout.create_batch(db, name="b", source_type="csv", created_by=None)
    with pytest.raises(PayloadInvalid):
        prescout.add_entry(
            db,
            batch_id=batch.id,
            team_number=254,
            season_year=2026,
            data={"drivetrain": "swerve", "extra": 1},
            scouter_id=None,
        )


def test_prescout_module_never_references_scouting_entries():
    src = pyinspect.getsource(prescout)
    assert "ScoutingEntry" not in src and "scouting_entries" not in src


def test_scouting_entry_model_is_distinct_table():
    assert ScoutingEntry.__tablename__ == "scouting_entries"
