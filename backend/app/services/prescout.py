"""Pre-scout write paths. These touch ONLY prescout_* tables (never scouting_entries)."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.models import PrescoutBatch, PrescoutEntry, Season
from app.services.schema_validate import validate_payload


def create_batch(
    db: Session, *, name: str, source_type: str, created_by: int | None,
    target_event_key: str | None = None, source_file: str | None = None,
) -> PrescoutBatch:
    batch = PrescoutBatch(
        name=name, source_type=source_type, created_by=created_by,
        target_event_key=target_event_key, source_file=source_file,
    )
    db.add(batch)
    db.flush()
    return batch


def add_entry(
    db: Session, *, batch_id: int, team_number: int, season_year: int, data: dict,
    scouter_id: int | None, source_event_key: str | None = None,
    source_match_key: str | None = None, video_url: str | None = None,
    notes: str | None = None, confidence: int | None = None,
) -> PrescoutEntry:
    season = db.get(Season, season_year)
    if season is None:
        raise LookupError(f"season {season_year} not seeded")
    validate_payload(season.prescout_schema, data)
    entry = PrescoutEntry(
        batch_id=batch_id, team_number=team_number, source_event_key=source_event_key,
        source_match_key=source_match_key, data=json.dumps(data), video_url=video_url,
        notes=notes, confidence=confidence, scouter_id=scouter_id,
    )
    db.add(entry)
    db.flush()
    return entry


