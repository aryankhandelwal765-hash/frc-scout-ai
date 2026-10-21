"""Event schedule, season schema and live scouting entries."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.audit import audit
from app.db import get_db
from app.deps import ANY_USER, STRATEGIST
from app.models import Event, Match, MatchTeam, ScoutingEntry, Season, User
from app.schemas import (
    EntryStatusIn,
    EventOut,
    MatchOut,
    MatchTeamOut,
    ScoutingBatchIn,
    ScoutingBatchItemOut,
    ScoutingBatchOut,
    ScoutingEntryIn,
    ScoutingEntryOut,
    SeasonOut,
)
from app.services.schema_validate import PayloadInvalid, validate_payload

router = APIRouter(tags=["scouting"])


def entry_out(e: ScoutingEntry) -> ScoutingEntryOut:
    return ScoutingEntryOut(
        id=e.id,
        match_key=e.match_key,
        team_number=e.team_number,
        scouter_id=e.scouter_id,
        data=json.loads(e.data),
        notes=e.notes,
        status=e.status,
        client_uuid=e.client_uuid,
        submitted_at=e.submitted_at,
    )


@router.get("/events", response_model=list[EventOut])
def list_events(db: Session = Depends(get_db), _u: User = Depends(ANY_USER)):
    return db.scalars(select(Event).order_by(Event.start_date.desc())).all()


@router.get("/events/{event_key}/matches", response_model=list[MatchOut], responses={404: {}})
def event_matches(
    event_key: str = Path(min_length=1, max_length=64),
    db: Session = Depends(get_db),
    _u: User = Depends(ANY_USER),
):
    if db.get(Event, event_key) is None:
        raise HTTPException(404, "Event not found")
    matches = db.scalars(
        select(Match)
        .where(Match.event_key == event_key)
        .order_by(Match.comp_level, Match.set_number, Match.match_number)
    ).all()
    teams: dict[str, list[MatchTeamOut]] = {}
    for mt in db.scalars(
        select(MatchTeam)
        .join(Match, Match.key == MatchTeam.match_key)
        .where(Match.event_key == event_key)
    ):
        teams.setdefault(mt.match_key, []).append(
            MatchTeamOut(team_number=mt.team_number, alliance=mt.alliance, slot=mt.slot)
        )
    return [
        MatchOut(
            key=m.key,
            comp_level=m.comp_level,
            set_number=m.set_number,
            match_number=m.match_number,
            red_score=m.red_score,
            blue_score=m.blue_score,
            played=bool(m.played),
            teams=sorted(teams.get(m.key, []), key=lambda t: (t.alliance, t.slot or 0)),
        )
        for m in matches
    ]


@router.get("/seasons/{year}", response_model=SeasonOut, responses={404: {}})
def get_season(
    year: int = Path(ge=1992, le=2100), db: Session = Depends(get_db), _u=Depends(ANY_USER)
):
    s = db.get(Season, year)
    if s is None:
        raise HTTPException(404, "Season not found")
    return SeasonOut(
        year=s.year,
        game_name=s.game_name,
        scouting_schema=json.loads(s.scouting_schema),
        prescout_schema=json.loads(s.prescout_schema) if s.prescout_schema else None,
    )


def _check_and_build(db: Session, user: User, body: ScoutingEntryIn) -> ScoutingEntry:
    """Validate one submission against the match roster and the season schema."""
    match = db.get(Match, body.match_key)
    if match is None:
        raise PayloadInvalid(["unknown match_key"])
    if db.get(MatchTeam, (body.match_key, body.team_number)) is None:
        raise PayloadInvalid([f"team {body.team_number} is not in match {body.match_key}"])
    event = db.get(Event, match.event_key)
    season = db.get(Season, event.season) if event else None
    validate_payload(season.scouting_schema if season else None, body.data)
    return ScoutingEntry(
        match_key=body.match_key,
        team_number=body.team_number,
        scouter_id=user.id,
        data=json.dumps(body.data),
        notes=body.notes,
        client_uuid=body.client_uuid,
    )


@router.post(
    "/scouting/entries",
    response_model=ScoutingEntryOut,
    status_code=201,
    responses={200: {"model": ScoutingEntryOut}, 422: {}},
)
def submit_entry(
    body: ScoutingEntryIn,
    response: Response,
    db: Session = Depends(get_db),
    user: User = Depends(ANY_USER),
):
    if body.client_uuid:
        dup = db.scalar(select(ScoutingEntry).where(ScoutingEntry.client_uuid == body.client_uuid))
        if dup:
            if dup.scouter_id != user.id:
                raise HTTPException(409, "client_uuid already used")
            response.status_code = 200
            return entry_out(dup)
    try:
        entry = _check_and_build(db, user, body)
    except PayloadInvalid as e:
        raise HTTPException(422, e.errors) from None
    db.add(entry)
    db.flush()
    audit(db, user.id, "scout_submit", f"entry={entry.id} match={entry.match_key}")
    db.commit()
    return entry_out(entry)


@router.post("/scouting/entries/batch", response_model=ScoutingBatchOut)
def submit_batch(
    body: ScoutingBatchIn, db: Session = Depends(get_db), user: User = Depends(ANY_USER)
):
    """Offline sync: one transaction, deduped on client_uuid; invalid items are reported."""
    items: list[ScoutingBatchItemOut] = []
    pending: list[tuple[int, ScoutingEntry]] = []
    seen: set[str] = set()
    for i, e in enumerate(body.entries):
        if e.client_uuid:
            dup = db.scalar(select(ScoutingEntry).where(ScoutingEntry.client_uuid == e.client_uuid))
            if (dup and dup.scouter_id == user.id) or e.client_uuid in seen:
                items.append(
                    ScoutingBatchItemOut(
                        index=i,
                        client_uuid=e.client_uuid,
                        result="duplicate",
                        id=dup.id if dup else None,
                    )
                )
                continue
            if dup:
                items.append(
                    ScoutingBatchItemOut(
                        index=i,
                        client_uuid=e.client_uuid,
                        result="invalid",
                        errors=["client_uuid already used"],
                    )
                )
                continue
            seen.add(e.client_uuid)
        try:
            pending.append((i, _check_and_build(db, user, e)))
        except PayloadInvalid as err:
            items.append(
                ScoutingBatchItemOut(
                    index=i, client_uuid=e.client_uuid, result="invalid", errors=err.errors
                )
            )
    try:
        for _i, row in pending:
            db.add(row)
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "Conflicting client_uuid in batch") from None
    for i, row in pending:
        items.append(
            ScoutingBatchItemOut(index=i, client_uuid=row.client_uuid, result="created", id=row.id)
        )
    audit(db, user.id, "scout_batch", f"created={len(pending)} total={len(body.entries)}")
    db.commit()
    return ScoutingBatchOut(items=sorted(items, key=lambda x: x.index))


@router.get("/scouting/entries/mine", response_model=list[ScoutingEntryOut])
def my_entries(
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    user: User = Depends(ANY_USER),
):
    rows = db.scalars(
        select(ScoutingEntry)
        .where(ScoutingEntry.scouter_id == user.id)
        .order_by(ScoutingEntry.id.desc())
        .limit(limit)
    )
    return [entry_out(e) for e in rows]


@router.get("/scouting/entries", response_model=list[ScoutingEntryOut])
def list_entries(
    match_key: str | None = Query(None, max_length=64),
    team: int | None = Query(None, ge=1, le=99999),
    status: str | None = Query(None, pattern="^(ok|flagged|rejected)$"),
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
    _s: User = Depends(STRATEGIST),
):
    q = select(ScoutingEntry).order_by(ScoutingEntry.id.desc()).limit(limit)
    if match_key:
        q = q.where(ScoutingEntry.match_key == match_key)
    if team:
        q = q.where(ScoutingEntry.team_number == team)
    if status:
        q = q.where(ScoutingEntry.status == status)
    return [entry_out(e) for e in db.scalars(q)]


@router.patch(
    "/scouting/entries/{entry_id}/status", response_model=ScoutingEntryOut, responses={404: {}}
)
def set_status(
    body: EntryStatusIn,
    entry_id: int = Path(ge=1),
    db: Session = Depends(get_db),
    user: User = Depends(STRATEGIST),
):
    e = db.get(ScoutingEntry, entry_id)
    if e is None:
        raise HTTPException(404, "Entry not found")
    e.status = body.status
    audit(db, user.id, "scout_status", f"entry={e.id} status={body.status}")
    db.commit()
    return entry_out(e)
