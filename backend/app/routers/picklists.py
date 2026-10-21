"""Pick lists: a strategist ranks teams for an event."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.audit import audit
from app.db import get_db
from app.deps import STRATEGIST
from app.models import Event, FrcTeam, Picklist, PicklistItem, User
from app.schemas import OkOut, PicklistCreateIn, PicklistItemIO, PicklistItemsIn, PicklistOut

router = APIRouter(prefix="/picklists", tags=["picklists"])


def _out(db: Session, p: Picklist) -> PicklistOut:
    items = db.scalars(
        select(PicklistItem).where(PicklistItem.picklist_id == p.id).order_by(PicklistItem.rank)
    )
    return PicklistOut(
        id=p.id,
        event_key=p.event_key,
        name=p.name,
        created_by=p.created_by,
        created_at=p.created_at,
        items=[
            PicklistItemIO(team_number=i.team_number, rank=i.rank, tier=i.tier, note=i.note)
            for i in items
        ],
    )


@router.post("", response_model=PicklistOut, status_code=201, responses={404: {}})
def create(body: PicklistCreateIn, db: Session = Depends(get_db), user: User = Depends(STRATEGIST)):
    if db.get(Event, body.event_key) is None:
        raise HTTPException(404, "Event not found")
    p = Picklist(event_key=body.event_key, name=body.name, created_by=user.id)
    db.add(p)
    db.flush()
    audit(db, user.id, "picklist_create", f"picklist={p.id}")
    db.commit()
    return _out(db, p)


@router.get("", response_model=list[PicklistOut])
def list_(
    event_key: str | None = Query(None, max_length=64),
    db: Session = Depends(get_db),
    _s: User = Depends(STRATEGIST),
):
    q = select(Picklist).order_by(Picklist.id.desc())
    if event_key:
        q = q.where(Picklist.event_key == event_key)
    return [_out(db, p) for p in db.scalars(q)]


@router.get("/{picklist_id}", response_model=PicklistOut, responses={404: {}})
def get(picklist_id: int = Path(ge=1), db: Session = Depends(get_db), _s=Depends(STRATEGIST)):
    p = db.get(Picklist, picklist_id)
    if p is None:
        raise HTTPException(404, "Picklist not found")
    return _out(db, p)


@router.put("/{picklist_id}/items", response_model=PicklistOut, responses={404: {}, 422: {}})
def replace_items(
    body: PicklistItemsIn,
    picklist_id: int = Path(ge=1),
    db: Session = Depends(get_db),
    user: User = Depends(STRATEGIST),
):
    p = db.get(Picklist, picklist_id)
    if p is None:
        raise HTTPException(404, "Picklist not found")
    teams = [i.team_number for i in body.items]
    ranks = [i.rank for i in body.items]
    if len(set(teams)) != len(teams):
        raise HTTPException(422, "A team appears twice")
    if len(set(ranks)) != len(ranks):
        raise HTTPException(422, "Two teams share a rank")
    known = set(db.scalars(select(FrcTeam.number).where(FrcTeam.number.in_(teams))))
    missing = sorted(set(teams) - known)
    if missing:
        raise HTTPException(422, f"Unknown teams: {missing}")
    db.execute(delete(PicklistItem).where(PicklistItem.picklist_id == p.id))
    for i in body.items:
        db.add(
            PicklistItem(
                picklist_id=p.id, team_number=i.team_number, rank=i.rank, tier=i.tier, note=i.note
            )
        )
    audit(db, user.id, "picklist_items", f"picklist={p.id} n={len(body.items)}")
    db.commit()
    return _out(db, p)


@router.delete("/{picklist_id}", response_model=OkOut, responses={404: {}})
def delete_(
    picklist_id: int = Path(ge=1), db: Session = Depends(get_db), user: User = Depends(STRATEGIST)
):
    p = db.get(Picklist, picklist_id)
    if p is None:
        raise HTTPException(404, "Picklist not found")
    db.execute(delete(PicklistItem).where(PicklistItem.picklist_id == p.id))
    db.delete(p)
    audit(db, user.id, "picklist_delete", f"picklist={picklist_id}")
    db.commit()
    return OkOut()
