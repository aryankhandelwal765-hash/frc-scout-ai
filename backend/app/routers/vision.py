"""Vision-derived data: admin import, per-team rows, and the team summary."""

from __future__ import annotations

import json
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.audit import audit
from app.db import get_db
from app.deps import ADMIN, STRATEGIST
from app.models import (
    Event,
    FrcTeam,
    Match,
    MatchTeam,
    PitScouting,
    PrescoutEntry,
    ScoutingEntry,
    User,
    VideoTrack,
)
from app.schemas import TeamSummaryOut, VisionImportOut, VisionRowOut
from app.services import ratings, tba, vision_import

router = APIRouter(tags=["vision"])


def _means(rows: list[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for d in rows:
        for k, v in d.items():
            if isinstance(v, bool):
                acc[k].append(float(v))
            elif isinstance(v, int | float):
                acc[k].append(float(v))
    return {k: round(sum(v) / len(v), 2) for k, v in acc.items() if v}


@router.post(
    "/admin/import/vision/{event_key}",
    response_model=VisionImportOut,
    responses={404: {}, 422: {}},
)
def import_vision(
    request: Request,
    event_key: str = Path(pattern=tba.EVENT_KEY_RE.pattern),
    db: Session = Depends(get_db),
    admin: User = Depends(ADMIN),
):
    if db.get(Event, event_key) is None:
        raise HTTPException(404, "Event not found; import it from TBA first")
    path = request.app.state.settings.vision_scout_table
    try:
        matches, rows = vision_import.import_scout_table(db, path, event_key)
    except FileNotFoundError:
        raise HTTPException(422, "Vision scout table not found on the server") from None
    audit(db, admin.id, "vision_import", f"event={event_key} rows={rows}")
    db.commit()
    return VisionImportOut(event_key=event_key, matches=matches, rows=rows)


@router.get("/vision/team/{team}", response_model=list[VisionRowOut])
def vision_team(
    team: int = Path(ge=1, le=99999),
    db: Session = Depends(get_db),
    _s: User = Depends(STRATEGIST),
):
    rows = db.scalars(
        select(VideoTrack).where(VideoTrack.team_number == team).order_by(VideoTrack.match_key)
    )
    return [
        VisionRowOut(match_key=r.match_key or "", team_number=team, data=json.loads(r.data))
        for r in rows
    ]


@router.get(
    "/events/{event_key}/teams/{team}/summary", response_model=TeamSummaryOut, responses={404: {}}
)
def team_summary(
    request: Request,
    event_key: str = Path(min_length=1, max_length=64),
    team: int = Path(ge=1, le=99999),
    db: Session = Depends(get_db),
    _s: User = Depends(STRATEGIST),
):
    if db.get(Event, event_key) is None:
        raise HTTPException(404, "Event not found")
    t = db.get(FrcTeam, team)
    in_event = select(Match.key).where(Match.event_key == event_key).scalar_subquery()
    scouted = [
        json.loads(e.data)
        for e in db.scalars(
            select(ScoutingEntry).where(
                ScoutingEntry.team_number == team,
                ScoutingEntry.match_key.in_(in_event),
                ScoutingEntry.status == "ok",
            )
        )
    ]
    vision = [
        json.loads(v.data)
        for v in db.scalars(
            select(VideoTrack).where(
                VideoTrack.team_number == team, VideoTrack.match_key.in_(in_event)
            )
        )
    ]
    if t is None and not scouted and not vision:
        raise HTTPException(404, "Team not found")
    played = db.scalar(
        select(func.count())
        .select_from(MatchTeam)
        .where(MatchTeam.team_number == team, MatchTeam.match_key.in_(in_event))
    )
    del played  # kept out of the response; scouting/vision counts are what a strategist acts on
    rating, rank = ratings.rating_and_rank(request.app.state.settings.history_ratings, team)
    return TeamSummaryOut(
        team_number=team,
        name=t.name if t else None,
        event_key=event_key,
        matches_scouted=len(scouted),
        scouting_means=_means(scouted),
        vision_matches=len(vision),
        vision_means=_means(vision),
        prescout_entries=db.scalar(
            select(func.count()).select_from(PrescoutEntry).where(PrescoutEntry.team_number == team)
        ),
        history_rating=rating,
        history_rank=rank,
        pit_entries=db.scalar(
            select(func.count())
            .select_from(PitScouting)
            .where(PitScouting.team_number == team, PitScouting.event_key == event_key)
        ),
    )
