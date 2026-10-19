"""TBA v3 importer: events, teams, matches, match_teams. Idempotent upserts."""

from __future__ import annotations

import re

import httpx
from sqlalchemy import delete
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from app.models import Event, FrcTeam, Match, MatchTeam, Season
from app.schemas import ImportSummaryOut

EVENT_KEY_RE = re.compile(r"^[0-9]{4}[a-z0-9]{2,30}$")


class TbaError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _get(client: httpx.Client, api_key: str, path: str):
    if not api_key:
        raise TbaError(503, "TBA API key not configured (set TBA_API_KEY)")
    try:
        r = client.get(path, headers={"X-TBA-Auth-Key": api_key})
    except httpx.HTTPError as e:
        raise TbaError(502, f"TBA unreachable: {type(e).__name__}") from None
    if r.status_code == 404:
        raise TbaError(404, f"TBA has no data for {path}")
    if r.status_code in (401, 403):
        raise TbaError(502, "TBA rejected the API key")
    if r.status_code != 200:
        raise TbaError(502, f"TBA returned HTTP {r.status_code}")
    return r.json()


def _team_num(team_key: str) -> int:
    return int(team_key.removeprefix("frc"))


def _score(v) -> int | None:
    return v if isinstance(v, int) and v >= 0 else None


def import_event(
    db: Session, client: httpx.Client, api_key: str, event_key: str
) -> ImportSummaryOut:
    if not EVENT_KEY_RE.match(event_key):
        raise TbaError(422, "Invalid event key")
    year = int(event_key[:4])
    if db.get(Season, year) is None:
        raise TbaError(409, f"Season {year} is not seeded; run `python -m app.cli seed-season`")

    ev = _get(client, api_key, f"/event/{event_key}")
    teams = _get(client, api_key, f"/event/{event_key}/teams/simple")
    matches = _get(client, api_key, f"/event/{event_key}/matches/simple")

    def upsert(model, values: dict, pk: list[str]):
        stmt = insert(model).values(**values)
        update_cols = {k: v for k, v in values.items() if k not in pk}
        if update_cols:
            stmt = stmt.on_conflict_do_update(index_elements=pk, set_=update_cols)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=pk)
        db.execute(stmt)

    upsert(
        Event,
        {
            "key": event_key,
            "season": year,
            "name": ev.get("name"),
            "start_date": ev.get("start_date"),
            "end_date": ev.get("end_date"),
        },
        ["key"],
    )
    for t in teams:
        upsert(
            FrcTeam,
            {
                "number": t["team_number"],
                "name": t.get("nickname"),
                "city": t.get("city"),
                "state_prov": t.get("state_prov"),
            },
            ["number"],
        )

    mt_count = 0
    for m in matches:
        red, blue = m["alliances"]["red"], m["alliances"]["blue"]
        rs, bs = _score(red.get("score")), _score(blue.get("score"))
        upsert(
            Match,
            {
                "key": m["key"],
                "event_key": event_key,
                "comp_level": m.get("comp_level"),
                "set_number": m.get("set_number"),
                "match_number": m.get("match_number"),
                "red_score": rs,
                "blue_score": bs,
                "played": int(rs is not None and bs is not None),
            },
            ["key"],
        )
        # Schedule can change (surrogates, replays): rebuild this match's rows.
        db.execute(delete(MatchTeam).where(MatchTeam.match_key == m["key"]))
        for alliance, side in (("red", red), ("blue", blue)):
            for slot, tk in enumerate(side.get("team_keys", []), start=1):
                num = _team_num(tk)
                db.execute(insert(FrcTeam).values(number=num).on_conflict_do_nothing())
                db.add(MatchTeam(match_key=m["key"], team_number=num, alliance=alliance, slot=slot))
                mt_count += 1
        db.flush()

    return ImportSummaryOut(
        event_key=event_key,
        season=year,
        teams=len(teams),
        matches=len(matches),
        match_teams=mt_count,
    )
