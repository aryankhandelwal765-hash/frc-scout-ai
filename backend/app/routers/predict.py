"""Match prediction from the ten-season ratings."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.deps import STRATEGIST
from app.models import User
from app.schemas import PredictOut
from app.services import ratings

router = APIRouter(tags=["predict"])


def _teams(raw: str) -> list[int]:
    try:
        teams = [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(422, "Teams must be comma-separated numbers") from None
    if not 1 <= len(teams) <= 4 or any(not 1 <= t <= 99999 for t in teams):
        raise HTTPException(422, "Each alliance needs 1-4 valid team numbers")
    return teams


@router.get("/predict", response_model=PredictOut, responses={422: {}, 503: {}})
def predict(
    request: Request,
    red: str = Query(max_length=40),
    blue: str = Query(max_length=40),
    _s: User = Depends(STRATEGIST),
):
    out = ratings.win_prob(request.app.state.settings.history_ratings, _teams(red), _teams(blue))
    if out is None:
        raise HTTPException(503, "Ratings file not built yet")
    return out
