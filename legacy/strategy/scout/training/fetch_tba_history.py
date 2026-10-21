"""
Download every played match of past FRC seasons from The Blue Alliance, resumably.

    .venv/bin/python -m scout.training.fetch_tba_history [--years 2016 2017 ...]

Writes scout/data/history/<year>/<event_key>.json (gitignored): one compact list of
{key, level, time, red:{teams,score}, blue:{teams,score}} per event. Only regular, district,
district-championship and championship events are kept, not offseason or preseason ones,
because those are the matches teams actually played for ranking. An event already on disk is
skipped, so a rerun after an interruption continues. The TBA key comes from scout.config
(TBA_AUTH_KEY) and is never printed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from scout.config import TBA_BASE, get_tba_key

HISTORY = Path(__file__).resolve().parent.parent / "data" / "history"
KEEP_TYPES = {
    0,
    1,
    2,
    3,
    4,
    5,
}  # regional, district, district champ, CMP division, CMP final, DCMP division
DEFAULT_YEARS = list(
    range(2016, 2027)
)  # 2020 is partial, 2021 has no events; both handled by emptiness


def get(session, path, retries=4):
    for attempt in range(retries):
        r = session.get(TBA_BASE + path, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2**attempt)
            continue
        if r.status_code == 404:
            return []
        r.raise_for_status()
    raise RuntimeError(f"TBA {path}: gave up after {retries} tries")


def compact(m):
    al = m.get("alliances", {})
    red, blue = al.get("red", {}), al.get("blue", {})
    if red.get("score", -1) < 0 or blue.get("score", -1) < 0:
        return None
    return {
        "key": m["key"],
        "level": m.get("comp_level"),
        "time": m.get("actual_time") or m.get("time") or m.get("predicted_time"),
        "red": {"teams": red.get("team_keys", []), "score": red["score"]},
        "blue": {"teams": blue.get("team_keys", []), "score": blue["score"]},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--years", nargs="*", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--pause", type=float, default=0.15)
    a = ap.parse_args()

    key = get_tba_key()
    if not key:
        sys.exit("No TBA key: set TBA_AUTH_KEY")
    s = requests.Session()
    s.headers["X-TBA-Auth-Key"] = key

    for year in a.years:
        events = [e for e in get(s, f"/events/{year}/simple") if e.get("event_type") in KEEP_TYPES]
        out = HISTORY / str(year)
        out.mkdir(parents=True, exist_ok=True)
        got = new = 0
        for e in sorted(events, key=lambda e: e["key"]):
            path = out / f"{e['key']}.json"
            if path.exists():
                got += 1
                continue
            matches = [c for m in get(s, f"/event/{e['key']}/matches/simple") if (c := compact(m))]
            path.write_text(json.dumps(matches, separators=(",", ":")))
            new += 1
            time.sleep(a.pause)
        print(f"{year}: {len(events)} events ({new} downloaded, {got} already on disk)", flush=True)


if __name__ == "__main__":
    main()
