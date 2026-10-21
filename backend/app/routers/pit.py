"""Pit scouting. The season has no pit schema, so the payload is a size-limited JSON object."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.audit import audit
from app.db import get_db
from app.deps import ANY_USER, STRATEGIST
from app.models import Event, FrcTeam, PitScouting, User
from app.schemas import PitIn, PitOut

router = APIRouter(prefix="/pit", tags=["pit"])
MAX_PIT_BYTES = 20_000


def _out(p: PitScouting) -> PitOut:
    return PitOut(
        id=p.id,
        event_key=p.event_key,
        team_number=p.team_number,
        scouter_id=p.scouter_id,
        data=json.loads(p.data),
        submitted_at=p.submitted_at,
    )


@router.post("", response_model=PitOut, status_code=201, responses={404: {}, 422: {}})
def submit(body: PitIn, db: Session = Depends(get_db), user: User = Depends(ANY_USER)):
    raw = json.dumps(body.data)
    if len(raw) > MAX_PIT_BYTES:
        raise HTTPException(422, "Pit data too large")
    if db.get(Event, body.event_key) is None:
        raise HTTPException(404, "Event not found")
    if db.get(FrcTeam, body.team_number) is None:
        raise HTTPException(404, "Team not found")
    p = PitScouting(
        event_key=body.event_key, team_number=body.team_number, scouter_id=user.id, data=raw
    )
    db.add(p)
    db.flush()
    audit(db, user.id, "pit_submit", f"pit={p.id} team={body.team_number}")
    db.commit()
    return _out(p)


@router.get("", response_model=list[PitOut])
def list_(
    event_key: str | None = Query(None, max_length=64),
    team: int | None = Query(None, ge=1, le=99999),
    db: Session = Depends(get_db),
    _s: User = Depends(STRATEGIST),
):
    q = select(PitScouting).order_by(PitScouting.id.desc()).limit(500)
    if event_key:
        q = q.where(PitScouting.event_key == event_key)
    if team:
        q = q.where(PitScouting.team_number == team)
    return [_out(p) for p in db.scalars(q)]
