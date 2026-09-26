"""Paths, model names and the TBA key loader. Secrets come from the environment."""

import os
from pathlib import Path

try:  # optional: load a local .env when python-dotenv is available
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a convenience, not a requirement
    load_dotenv = None

# =========================================================
# ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Repo root is strategy/scout -> strategy -> <root>.
REPO_ROOT = BASE_DIR.parent.parent

if load_dotenv is not None:
    # Never overrides a variable already set in the real environment.
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(BASE_DIR.parent / ".env", override=False)

# =========================================================
# DATA
# =========================================================

DATA_DIR = BASE_DIR / "data"

RAW_MATCHES_DIR = DATA_DIR / "raw_matches"

PROCESSED_DIR = DATA_DIR / "processed"

SCOUTING_DIR = DATA_DIR / "scouting"

STATBOTICS_DIR = DATA_DIR / "statbotics"

# =========================================================
# MODELS
# =========================================================

MODELS_DIR = BASE_DIR / "models"

# =====================================================
# STATBOTICS
# =====================================================

STATBOTICS_FILE = STATBOTICS_DIR / "all_2026_team_data.json"

# =====================================================
# MODEL OUTPUTS
# =====================================================

MATCHUP_MODEL_DIR = BASE_DIR / "models"

MATCHUP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# OLLAMA
# =====================================================

# Base URL of the Ollama server. OLLAMA_HOST is the documented name;
# OLLAMA_BASE is accepted as a legacy alias so existing compose files keep
# working.
OLLAMA_BASE = (
    os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_BASE") or "http://localhost:11434"
).rstrip("/")

# Strategy-captain qualitative synthesis (the reasoning layer).
# Was qwen3.6:35b-a3b (~70s). qwen3:8b is ~5-10s w/ good enough reasoning.
SYNTHESIS_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

# Small, fast model for lightweight/coding tasks.
CODE_MODEL = os.environ.get("OLLAMA_CODE_MODEL", "qwen2.5:3b")

# Per-robot ("bot-by-bot") analysis — runs once per team, so it uses the same
# small model and disables thinking at call time for speed.
BOT_MODEL = os.environ.get("OLLAMA_BOT_MODEL", SYNTHESIS_MODEL)

# =====================================================
# DATA PROVIDER
# =====================================================

# Which backend supplies team stats: "tba" (The Blue Alliance) or "statbotics".
# Statbotics' data API has been down since 2026-07-09 (their issues #414/#418),
# so TBA is the default. Flip back to "statbotics" once it recovers.
DATA_PROVIDER = os.environ.get("DATA_PROVIDER", "tba")

# The Blue Alliance API v3.
TBA_BASE = "https://www.thebluealliance.com/api/v3"


def get_tba_key() -> str | None:
    """Read the TBA read API key from the environment.

    Source of truth is the ``TBA_AUTH_KEY`` environment variable. A ``.env``
    file at the repo root (or in ``strategy/``) is loaded into the environment
    at import time when python-dotenv is installed. Returns None if no key is
    configured.
    """
    key = os.environ.get("TBA_AUTH_KEY")
    return key.strip() if key else None
