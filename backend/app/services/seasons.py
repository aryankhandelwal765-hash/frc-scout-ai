"""Season seeding. The bundled schema is a generic EXAMPLE, not real game data."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.models import Season
from app.services.schema_validate import compile_schema

EXAMPLE_YEAR = 2026
EXAMPLE_GAME_NAME = "EXAMPLE – replace me"

EXAMPLE_SCOUTING_SCHEMA = {
    "$comment": "EXAMPLE schema with generic fields. Replace with the real game's fields.",
    "type": "object",
    "properties": {
        "auto_scored": {"type": "integer", "minimum": 0, "maximum": 100, "x-weight": 1.0},
        "teleop_scored": {"type": "integer", "minimum": 0, "maximum": 200, "x-weight": 1.0},
        "played_defense": {"type": "boolean"},
        "driver_skill": {"type": "integer", "minimum": 1, "maximum": 5},
        "broke_down": {"type": "boolean"},
    },
    "required": ["auto_scored", "teleop_scored"],
    "additionalProperties": False,
}

EXAMPLE_PRESCOUT_SCHEMA = {
    "$comment": "EXAMPLE pre-scout schema. Replace with real fields.",
    "type": "object",
    "properties": {
        "drivetrain": {"type": "string", "enum": ["swerve", "tank", "other", "unknown"]},
        "avg_scored": {"type": "number", "minimum": 0},
        "reliability": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["drivetrain"],
    "additionalProperties": False,
}


def seed_example_season(db: Session, *, force: bool = False) -> tuple[Season, bool]:
    """Insert the example season if absent. Returns (season, created)."""
    compile_schema(EXAMPLE_SCOUTING_SCHEMA)
    compile_schema(EXAMPLE_PRESCOUT_SCHEMA)
    existing = db.get(Season, EXAMPLE_YEAR)
    if existing and not force:
        return existing, False
    if existing:
        db.delete(existing)
        db.flush()
    season = Season(
        year=EXAMPLE_YEAR,
        game_name=EXAMPLE_GAME_NAME,
        scouting_schema=json.dumps(EXAMPLE_SCOUTING_SCHEMA),
        prescout_schema=json.dumps(EXAMPLE_PRESCOUT_SCHEMA),
    )
    db.add(season)
    db.commit()
    return season, True
