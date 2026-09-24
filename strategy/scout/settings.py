"""User settings with on-disk persistence.

Streamlit session_state resets on restart, so the user's team number is stored in
a small JSON file (BASE_DIR/user_settings.json) that survives restarts.
"""

import json

from scout.config import BASE_DIR

_SETTINGS_PATH = BASE_DIR / "user_settings.json"


def load_settings() -> dict:
    if _SETTINGS_PATH.exists():
        try:
            return json.loads(_SETTINGS_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_settings(data: dict) -> None:
    _SETTINGS_PATH.write_text(json.dumps(data, indent=2))


def get_my_team() -> int | None:
    val = load_settings().get("my_team")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def set_my_team(team: int | None) -> None:
    data = load_settings()
    if team is None:
        data.pop("my_team", None)
    else:
        data["my_team"] = int(team)
    save_settings(data)
