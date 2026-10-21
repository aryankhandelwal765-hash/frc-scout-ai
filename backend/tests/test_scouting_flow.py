"""Scouting, pit, picklist, prescout, vision import and summary behaviour."""

from __future__ import annotations

import pytest

from app.models import Event, FrcTeam, Match, MatchTeam
from app.services import seasons

EVENT = "2026test"
MATCH = "2026test_qm1"
GOOD = {"auto_scored": 10, "teleop_scored": 40}


@pytest.fixture
def seeded(app):
    with app.state.session_factory() as s:
        seasons.seed_example_season(s)
        s.add(Event(key=EVENT, season=2026, name="Test"))
        for n in (254, 1678, 971, 118, 148, 111, 9999):
            s.add(FrcTeam(number=n, name=f"Team {n}"))
        s.add(Match(key=MATCH, event_key=EVENT, comp_level="qm", set_number=1, match_number=1))
        s.flush()
        for slot, (n, al) in enumerate(
            [(254, "red"), (1678, "red"), (971, "red"), (118, "blue"), (148, "blue"), (111, "blue")]
        ):
            s.add(MatchTeam(match_key=MATCH, team_number=n, alliance=al, slot=slot % 3))
        s.commit()


def entry(**kw):
    return {"match_key": MATCH, "team_number": 254, "data": GOOD, **kw}


def test_scout_submits_valid_entry(seeded, make_user, login):
    make_user("sam", "scout")
    c = login("sam")
    r = c.post("/scouting/entries", json=entry(client_uuid="uuid-aaaa-1111"))
    assert r.status_code == 201, r.text
    assert r.json()["data"] == GOOD and r.json()["status"] == "ok"


def test_unknown_field_is_rejected(seeded, make_user, login):
    make_user("sam", "scout")
    r = login("sam").post("/scouting/entries", json=entry(data={**GOOD, "bogus": 1}))
    assert r.status_code == 422


def test_team_must_be_in_the_match(seeded, make_user, login):
    make_user("sam", "scout")
    r = login("sam").post("/scouting/entries", json=entry(team_number=9999))
    assert r.status_code == 422


def test_resubmit_with_same_uuid_returns_the_original(seeded, make_user, login):
    make_user("sam", "scout")
    c = login("sam")
    a = c.post("/scouting/entries", json=entry(client_uuid="uuid-bbbb-2222"))
    b = c.post("/scouting/entries", json=entry(client_uuid="uuid-bbbb-2222"))
    assert (a.status_code, b.status_code) == (201, 200)
    assert a.json()["id"] == b.json()["id"]
    assert len(c.get("/scouting/entries/mine").json()) == 1


def test_another_scouter_cannot_reuse_a_uuid(seeded, make_user, login):
    make_user("sam", "scout")
    make_user("tia", "scout")
    login("sam").post("/scouting/entries", json=entry(client_uuid="uuid-cccc-3333"))
    r = login("tia").post("/scouting/entries", json=entry(client_uuid="uuid-cccc-3333"))
    assert r.status_code == 409


def test_batch_reports_each_item(seeded, make_user, login):
    make_user("sam", "scout")
    c = login("sam")
    c.post("/scouting/entries", json=entry(client_uuid="uuid-dddd-4444"))
    body = {
        "entries": [
            entry(client_uuid="uuid-dddd-4444"),  # duplicate
            entry(team_number=1678, client_uuid="uuid-eeee-5555"),  # new
            entry(team_number=9999),  # not in match
            entry(team_number=971, data={"auto_scored": "x", "teleop_scored": 1}),  # bad data
            entry(team_number=118, client_uuid="uuid-eeee-5555"),  # repeat inside the batch
        ]
    }
    r = c.post("/scouting/entries/batch", json=body)
    assert r.status_code == 200, r.text
    assert [i["result"] for i in r.json()["items"]] == [
        "duplicate",
        "created",
        "invalid",
        "invalid",
        "duplicate",
    ]
    assert len(c.get("/scouting/entries/mine").json()) == 2


def test_scouts_cannot_read_everyone_but_strategists_can(seeded, make_user, login):
    make_user("sam", "scout")
    make_user("stan", "strategist")
    login("sam").post("/scouting/entries", json=entry())
    assert login("sam").get("/scouting/entries").status_code == 403
    rows = login("stan").get("/scouting/entries", params={"team": 254}).json()
    assert len(rows) == 1


def test_strategist_can_flag_and_flagged_rows_leave_the_summary(seeded, make_user, login):
    make_user("sam", "scout")
    make_user("stan", "strategist")
    eid = login("sam").post("/scouting/entries", json=entry()).json()["id"]
    st = login("stan")
    s1 = st.get(f"/events/{EVENT}/teams/254/summary").json()
    assert s1["matches_scouted"] == 1 and s1["scouting_means"]["teleop_scored"] == 40
    assert (
        st.patch(f"/scouting/entries/{eid}/status", json={"status": "flagged"}).status_code == 200
    )
    assert st.get(f"/events/{EVENT}/teams/254/summary").json()["matches_scouted"] == 0
    assert (
        login("sam").patch(f"/scouting/entries/{eid}/status", json={"status": "ok"}).status_code
        == 403
    )


def test_schedule_lists_match_teams(seeded, make_user, login):
    make_user("sam", "scout")
    ms = login("sam").get(f"/events/{EVENT}/matches").json()
    assert ms[0]["key"] == MATCH and len(ms[0]["teams"]) == 6


def test_season_schema_is_served(seeded, make_user, login):
    make_user("sam", "scout")
    r = login("sam").get("/seasons/2026")
    assert r.status_code == 200 and "auto_scored" in r.json()["scouting_schema"]["properties"]


def test_pit_entry_roundtrip(seeded, make_user, login):
    make_user("sam", "scout")
    make_user("stan", "strategist")
    r = login("sam").post(
        "/pit", json={"event_key": EVENT, "team_number": 254, "data": {"drivetrain": "swerve"}}
    )
    assert r.status_code == 201
    assert len(login("stan").get("/pit", params={"team": 254}).json()) == 1


def test_picklist_replace_and_validation(seeded, make_user, login):
    make_user("stan", "strategist")
    c = login("stan")
    pid = c.post("/picklists", json={"event_key": EVENT, "name": "First"}).json()["id"]
    items = [{"team_number": 254, "rank": 1, "tier": "A"}, {"team_number": 1678, "rank": 2}]
    r = c.put(f"/picklists/{pid}/items", json={"items": items})
    assert r.status_code == 200 and [i["team_number"] for i in r.json()["items"]] == [254, 1678]
    dup = c.put(f"/picklists/{pid}/items", json={"items": [items[0], items[0]]})
    assert dup.status_code == 422
    unknown = c.put(f"/picklists/{pid}/items", json={"items": [{"team_number": 5, "rank": 1}]})
    assert unknown.status_code == 422
    assert c.delete(f"/picklists/{pid}").status_code == 200
    assert c.get(f"/picklists/{pid}").status_code == 404


def test_prescout_stays_out_of_live_scouting(seeded, make_user, login):
    make_user("sam", "scout")
    make_user("stan", "strategist")
    c = login("sam")
    b = c.post("/prescout/batches", json={"name": "Week 1", "target_event_key": EVENT}).json()
    r = c.post(
        f"/prescout/batches/{b['id']}/entries",
        json={"team_number": 254, "season_year": 2026, "data": {"drivetrain": "swerve"}},
    )
    assert r.status_code == 201, r.text
    st = login("stan")
    assert len(st.get("/prescout/entries", params={"team": 254}).json()) == 1
    assert st.get("/scouting/entries").json() == []
    bad = c.post(
        f"/prescout/batches/{b['id']}/entries",
        json={"team_number": 254, "season_year": 2026, "data": {"drivetrain": "hovercraft"}},
    )
    assert bad.status_code == 422


def test_scout_cannot_write_into_someone_elses_batch(seeded, make_user, login):
    make_user("sam", "scout")
    make_user("tia", "scout")
    b = login("sam").post("/prescout/batches", json={"name": "Mine"}).json()
    r = login("tia").post(
        f"/prescout/batches/{b['id']}/entries",
        json={"team_number": 254, "season_year": 2026, "data": {"drivetrain": "swerve"}},
    )
    assert r.status_code == 403


def test_vision_import_and_summary(seeded, app, admin_client, make_user, login, tmp_path):
    csv = tmp_path / "scout_table.csv"
    csv.write_text(
        "match,team,alliance,auto_fuel,teleop_fuel,tele_bump,driver_skill,notes\n"
        f"{MATCH},254,red,12,40,2,,\n"
        f"{MATCH},1678,red,8,,0,5,human note\n"
        "2026other_qm1,254,red,99,99,9,,\n"
    )
    app.state.settings.vision_scout_table = str(csv)
    r = admin_client.post(f"/admin/import/vision/{EVENT}")
    assert r.status_code == 200, r.text
    assert r.json() == {"event_key": EVENT, "matches": 1, "rows": 2}
    make_user("stan", "strategist")
    st = login("stan")
    rows = st.get("/vision/team/1678").json()
    assert rows[0]["data"] == {
        "alliance": "red",
        "auto_fuel": 8,
        "tele_bump": 0,
    }  # blanks and human columns dropped
    s = st.get(f"/events/{EVENT}/teams/254/summary").json()
    assert s["vision_matches"] == 1 and s["vision_means"]["teleop_fuel"] == 40
    # re-import replaces, does not duplicate
    admin_client.post(f"/admin/import/vision/{EVENT}")
    assert len(st.get("/vision/team/254").json()) == 1


def test_vision_import_needs_the_table(seeded, app, admin_client):
    app.state.settings.vision_scout_table = "/nonexistent/scout_table.csv"
    assert admin_client.post(f"/admin/import/vision/{EVENT}").status_code == 422


def _ratings(app, tmp_path):
    import json

    p = tmp_path / "team_ratings.json"
    p.write_text(
        json.dumps({"logistic_scale": 400, "ratings": {"254": 1900, "1678": 1500, "971": 1100}})
    )
    app.state.settings.history_ratings = str(p)


def test_predict_uses_ratings_and_reports_unknown_teams(seeded, app, make_user, login, tmp_path):
    _ratings(app, tmp_path)
    make_user("stan", "strategist")
    c = login("stan")
    r = c.get("/predict", params={"red": "254,1678", "blue": "971,5"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["red_win_prob"] > 0.5 and body["unknown_teams"] == [5]
    assert abs(body["red_win_prob"] + body["blue_win_prob"] - 1) < 1e-9
    assert c.get("/predict", params={"red": "x", "blue": "1"}).status_code == 422


def test_predict_needs_a_built_ratings_file(seeded, app, make_user, login):
    app.state.settings.history_ratings = "/nonexistent/team_ratings.json"
    make_user("stan", "strategist")
    assert login("stan").get("/predict", params={"red": "1", "blue": "2"}).status_code == 503


def test_scouts_cannot_predict(seeded, make_user, login):
    make_user("sam", "scout")
    assert login("sam").get("/predict", params={"red": "1", "blue": "2"}).status_code == 403


def test_summary_carries_the_history_rating(seeded, app, make_user, login, tmp_path):
    _ratings(app, tmp_path)
    make_user("stan", "strategist")
    s = login("stan").get(f"/events/{EVENT}/teams/254/summary").json()
    assert s["history_rating"] == 1900 and s["history_rank"] == 1
