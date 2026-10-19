import httpx
import pytest
from sqlalchemy import func, select, text

from app.models import Event, FrcTeam, Match, MatchTeam
from app.services import seasons, tba

KEY = "2026test"


def fake_tba(matches=None):
    def alliance(score, keys):
        return {"score": score, "team_keys": keys}

    default_matches = [
        {
            "key": f"{KEY}_qm1",
            "comp_level": "qm",
            "set_number": 1,
            "match_number": 1,
            "alliances": {
                "red": alliance(50, ["frc1", "frc2", "frc3"]),
                "blue": alliance(40, ["frc4", "frc5", "frc99"]),
            },
        },
        {
            "key": f"{KEY}_qm2",
            "comp_level": "qm",
            "set_number": 1,
            "match_number": 2,
            "alliances": {
                "red": alliance(-1, ["frc1", "frc2", "frc3"]),
                "blue": alliance(-1, ["frc4", "frc5", "frc6"]),
            },
        },
    ]
    payload = {
        f"/event/{KEY}": {
            "key": KEY,
            "name": "Test Regional",
            "start_date": "2026-03-01",
            "end_date": "2026-03-03",
            "year": 2026,
        },
        f"/event/{KEY}/teams/simple": [
            {"team_number": n, "nickname": f"Team {n}", "city": "C", "state_prov": "S"}
            for n in (1, 2, 3, 4, 5, 6)
        ],
        f"/event/{KEY}/matches/simple": matches or default_matches,
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["X-TBA-Auth-Key"] == "fake-key"
        path = req.url.path
        if path in payload:
            return httpx.Response(200, json=payload[path])
        return httpx.Response(404)

    return handler


def counts(db):
    return {
        t: db.scalar(text(f"SELECT COUNT(*) FROM {t}"))
        for t in ("events", "frc_teams", "matches", "match_teams")
    }


def test_import_requires_seeded_season(admin_client, mock_tba):
    mock_tba["handler"] = fake_tba()
    r = admin_client.post(f"/admin/import/event/{KEY}")
    assert r.status_code == 409 and "seed-season" in r.json()["detail"]
    assert mock_tba["calls"] == []  # failed before touching the network


def test_import_and_rerun_is_idempotent(admin_client, mock_tba, db):
    seasons.seed_example_season(db)
    mock_tba["handler"] = fake_tba()
    r = admin_client.post(f"/admin/import/event/{KEY}")
    assert r.status_code == 200, r.text
    assert r.json() == {
        "event_key": KEY,
        "season": 2026,
        "teams": 6,
        "matches": 2,
        "match_teams": 12,
    }
    first = counts(db)
    assert first == {"events": 1, "frc_teams": 7, "matches": 2, "match_teams": 12}  # 99 auto-added

    # re-run with changed data: updates, no duplicates
    changed = fake_tba()
    mock_tba["handler"] = changed
    for _ in range(2):
        assert admin_client.post(f"/admin/import/event/{KEY}").status_code == 200
    db.expire_all()
    assert counts(db) == first

    m1 = db.get(Match, f"{KEY}_qm1")
    m2 = db.get(Match, f"{KEY}_qm2")
    assert (m1.red_score, m1.blue_score, m1.played) == (50, 40, 1)
    assert (m2.red_score, m2.blue_score, m2.played) == (None, None, 0)
    mt = db.get(MatchTeam, (f"{KEY}_qm1", 99))
    assert (mt.alliance, mt.slot) == ("blue", 3)
    assert db.get(Event, KEY).name == "Test Regional"
    assert db.get(FrcTeam, 1).name == "Team 1"


def test_rerun_picks_up_score_and_roster_changes(admin_client, mock_tba, db):
    seasons.seed_example_season(db)
    mock_tba["handler"] = fake_tba()
    admin_client.post(f"/admin/import/event/{KEY}")
    updated = [
        {
            "key": f"{KEY}_qm2",
            "comp_level": "qm",
            "set_number": 1,
            "match_number": 2,
            "alliances": {
                "red": {"score": 70, "team_keys": ["frc1", "frc2", "frc4"]},
                "blue": {"score": 60, "team_keys": ["frc3", "frc5", "frc6"]},
            },
        },
    ]
    mock_tba["handler"] = fake_tba(updated)
    assert admin_client.post(f"/admin/import/event/{KEY}").status_code == 200
    db.expire_all()
    m = db.get(Match, f"{KEY}_qm2")
    assert (m.red_score, m.played) == (70, 1)
    reds = db.scalars(
        select(MatchTeam.team_number).where(
            MatchTeam.match_key == f"{KEY}_qm2", MatchTeam.alliance == "red"
        )
    ).all()
    assert sorted(reds) == [1, 2, 4]
    assert (
        db.scalar(
            select(func.count()).select_from(MatchTeam).where(MatchTeam.match_key == f"{KEY}_qm2")
        )
        == 6
    )


def test_tba_failures_map_to_clean_errors(admin_client, mock_tba, db):
    seasons.seed_example_season(db)
    mock_tba["handler"] = lambda req: httpx.Response(404)
    assert admin_client.post(f"/admin/import/event/{KEY}").status_code == 404
    mock_tba["handler"] = lambda req: httpx.Response(500)
    assert admin_client.post(f"/admin/import/event/{KEY}").status_code == 502
    assert counts(db)["events"] == 0  # rolled back


def test_bad_event_key_rejected(admin_client, mock_tba):
    assert admin_client.post("/admin/import/event/not-a-key").status_code == 422
    assert admin_client.post("/admin/import/event/2026..").status_code in (404, 422)


def test_missing_api_key_is_503(app, admin_client, mock_tba, db):
    from pydantic import SecretStr

    seasons.seed_example_season(db)
    app.state.settings.tba_api_key = SecretStr("")
    mock_tba["handler"] = fake_tba()
    assert admin_client.post(f"/admin/import/event/{KEY}").status_code == 503


def test_import_requires_admin(make_user, login, mock_tba):
    make_user("st", "strategist")
    assert login("st").post(f"/admin/import/event/{KEY}").status_code == 403


def test_service_raises_tba_error_for_unseeded(db):
    with pytest.raises(tba.TbaError) as e:
        tba.import_event(db, httpx.Client(transport=httpx.MockTransport(fake_tba())), "k", KEY)
    assert e.value.status == 409
