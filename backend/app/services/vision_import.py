"""Load the vision pipeline's per-(match, team) table into video_tracks.

Source is vision/match_pipeline/scout_table.csv: one row per (match, team), with TBA-exact
alliance totals, the vision-apportioned per-robot split, and zone facts. Empty cells are dropped
rather than stored as zero, because "not measured" is not the same as "measured zero".
Re-importing a match replaces its rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import Match, VideoTrack

HUMAN_COLUMNS = {
    "failed_climb",
    "baseline",
    "defense",
    "tele_passing",
    "hp_accuracy",
    "reliability",
    "driver_skill",
    "notes",
}


def _cell(v: str):
    v = v.strip()
    if v == "":
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def import_scout_table(db: Session, path: str, event_key: str) -> tuple[int, int]:
    """Returns (matches, rows). Rows for matches not in the DB are skipped."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    known = set(
        db.scalars(
            Match.__table__.select()
            .with_only_columns(Match.key)
            .where(Match.event_key == event_key)
        )
    )
    by_match: dict[str, list[tuple[int, dict]]] = {}
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = row["match"]
            if key not in known:
                continue
            data = {
                k: c
                for k, v in row.items()
                if k not in ("match", "team") and k not in HUMAN_COLUMNS
                if (c := _cell(v)) is not None
            }
            team = _cell(row["team"])
            if isinstance(team, int):
                by_match.setdefault(key, []).append((team, data))
    rows = 0
    for key, items in by_match.items():
        db.execute(delete(VideoTrack).where(VideoTrack.match_key == key))
        for team, data in items:
            db.add(VideoTrack(match_key=key, team_number=team, data=json.dumps(data)))
            rows += 1
    db.flush()
    return len(by_match), rows
