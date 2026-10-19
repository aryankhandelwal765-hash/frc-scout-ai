"""Write the OpenAPI contract to backend/openapi.json (deterministic).

Run from backend/:  .venv/bin/python -m scripts.export_openapi
"""

import json
from pathlib import Path

from app.config import Settings
from app.main import create_app

OUT = Path(__file__).resolve().parent.parent / "openapi.json"


def build_spec() -> dict:
    # Throwaway in-memory settings; nothing here touches a database or a real key.
    settings = Settings(_env_file=None, env="dev", secret_key="x" * 40, cookie_secure=True,
                        database_url="sqlite://")
    return create_app(settings).openapi()


if __name__ == "__main__":
    OUT.write_text(json.dumps(build_spec(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
