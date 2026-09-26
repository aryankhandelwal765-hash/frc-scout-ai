"""
Thin The Blue Alliance (TBA) API v3 client.

Read-only. Needs an auth key (X-TBA-Auth-Key). Supply it via, in priority order:
  1. --key CLI arg on the scripts that use this
  2. TBA_AUTH_KEY environment variable
  3. a file named `.tba_key` next to this module (one line, the key)

Get a key (free) at https://www.thebluealliance.com/account -> "Read API Keys".
"""

import os
import ssl
import time
import json
import urllib.request
import urllib.error

BASE = "https://www.thebluealliance.com/api/v3"
_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tba_key")


def _ssl_context():
    """python.org framework Python ships no system CA bundle -> SSL verify fails.
    Use certifi's bundle (pulled in by `requests`, already in the venv)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL = _ssl_context()


def load_key(cli_key=None):
    if cli_key:
        return cli_key.strip()
    env = os.environ.get("TBA_AUTH_KEY")
    if env:
        return env.strip()
    if os.path.exists(_KEY_FILE):
        with open(_KEY_FILE) as f:
            k = f.read().strip()
            if k:
                return k
    raise SystemExit(
        "No TBA key. Set TBA_AUTH_KEY, pass --key, or write it to "
        f"{_KEY_FILE}\nGet one at https://www.thebluealliance.com/account"
    )


def get(path, key, retries=4):
    """GET {BASE}{path} with the auth header, JSON-decoded. Retries on 429/5xx."""
    url = BASE + path
    req = urllib.request.Request(
        url,
        headers={
            "X-TBA-Auth-Key": key,
            "User-Agent": "vision-tba-batch/1.0",
        },
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30, context=_SSL) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise SystemExit("TBA 401: bad/missing auth key.")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise


def event_matches(event_key, key):
    """Full match objects for an event (includes videos + score_breakdown)."""
    return get(f"/event/{event_key}/matches", key)


# canonical match ordering: quals, then playoffs by set/match
_LEVEL_ORDER = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}


def match_sort_key(m):
    return (
        _LEVEL_ORDER.get(m.get("comp_level"), 9),
        m.get("set_number") or 0,
        m.get("match_number") or 0,
    )


def youtube_key(match):
    """First YouTube video key on a match, or None."""
    for v in match.get("videos") or []:
        if v.get("type") == "youtube" and v.get("key"):
            return v["key"]
    return None
