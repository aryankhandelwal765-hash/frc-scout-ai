"""Pre-scout area. Writes go through services/prescout.py and touch only prescout_* tables."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.audit import audit
from app.db import get_db
from app.deps import ANY_USER, STRATEGIST
from app.models import Event, FrcTeam, PrescoutBatch, PrescoutEntry, User
from app.schemas import PrescoutBatchIn, PrescoutBatchOut, PrescoutEntryIn, PrescoutEntryOut
from app.services import prescout
from app.services.schema_validate import PayloadInvalid

router = APIRouter(prefix="/prescout", tags=["prescout"])


def _out(e: PrescoutEntry) -> PrescoutEntryOut:
    return PrescoutEntryOut(
        id=e.id,
        batch_id=e.batch_id,
        team_number=e.team_number,
        source_event_key=e.source_event_key,
        source_match_key=e.source_match_key,
        data=json.loads(e.data),
        video_url=e.video_url,
        notes=e.notes,
        confidence=e.confidence,
        scouter_id=e.scouter_id,
        created_at=e.created_at,
    )


@router.post("/batches", response_model=PrescoutBatchOut, status_code=201, responses={404: {}})
def create_batch(
    body: PrescoutBatchIn, db: Session = Depends(get_db), user: User = Depends(ANY_USER)
):
    if body.target_event_key and db.get(Event, body.target_event_key) is None:
        raise HTTPException(404, "Event not found")
    b = prescout.create_batch(
        db,
        name=body.name,
        source_type=body.source_type,
        created_by=user.id,
        target_event_key=body.target_event_key,
    )
    audit(db, user.id, "prescout_batch", f"batch={b.id}")
    db.commit()
    return b


@router.get("/batches", response_model=list[PrescoutBatchOut])
def list_batches(db: Session = Depends(get_db), _s: User = Depends(STRATEGIST)):
    return db.scalars(select(PrescoutBatch).order_by(PrescoutBatch.id.desc())).all()


@router.post(
    "/batches/{batch_id}/entries",
    response_model=PrescoutEntryOut,
    status_code=201,
    responses={404: {}, 422: {}},
)
def add_entry(
    body: PrescoutEntryIn,
    batch_id: int = Path(ge=1),
    db: Session = Depends(get_db),
    user: User = Depends(ANY_USER),
):
    batch = db.get(PrescoutBatch, batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found")
    if batch.created_by != user.id and user.role == "scout":
        raise HTTPException(403, "Not your batch")
    if db.get(FrcTeam, body.team_number) is None:
        raise HTTPException(404, "Team not found")
    try:
        e = prescout.add_entry(
            db,
            batch_id=batch_id,
            team_number=body.team_number,
            season_year=body.season_year,
            data=body.data,
            scouter_id=user.id,
            source_event_key=body.source_event_key,
            source_match_key=body.source_match_key,
            video_url=body.video_url,
            notes=body.notes,
            confidence=body.confidence,
        )
    except LookupError as err:
        raise HTTPException(404, str(err)) from None
    except PayloadInvalid as err:
        raise HTTPException(422, err.errors) from None
    audit(db, user.id, "prescout_entry", f"entry={e.id} team={e.team_number}")
    db.commit()
    return _out(e)


@router.get("/entries", response_model=list[PrescoutEntryOut])
def list_entries(
    team: int | None = Query(None, ge=1, le=99999),
    batch_id: int | None = Query(None, ge=1),
    db: Session = Depends(get_db),
    _s: User = Depends(STRATEGIST),
):
    q = select(PrescoutEntry).order_by(PrescoutEntry.id.desc()).limit(500)
    if team:
        q = q.where(PrescoutEntry.team_number == team)
    if batch_id:
        q = q.where(PrescoutEntry.batch_id == batch_id)
    return [_out(e) for e in db.scalars(q)]
