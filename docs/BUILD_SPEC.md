# FRC Scout AI: Build Spec

<aside>
🎯

Handoff doc for Claude Code. Team-only, password-protected scouting + strategy app. Max 6 scouters, so SQLite is the DB. Pre-scouting lives in its own tables and its own section of the app so it never contaminates live match data.

</aside>

## Decisions

- SQLite (WAL mode). 6 scouters max, one writer at a time is fine. Use Alembic batch mode because SQLite can't do most ALTERs.
- Single team, no multi-tenant org_id. Team number lives in app_settings.
- Per-user accounts, not one shared password. Reason: you can revoke one person, and every scouting entry is attributable so bad data can be traced. Signup is by invite code only.
- Roles: scout (submit data), strategist (view everything, build picklists, run AI), admin (users, imports, settings).
- Game-specific fields live in a JSON text column validated against a per-season schema (seasons.scouting_schema). Game changes every year, so no table rewrites.
- Pre-scout data is separate from live scouting data. The ML model blends them with explicit weights, never merges rows.
- Stack: FastAPI, SQLAlchemy 2.x, Alembic, Pydantic, argon2-cffi, pytest. Ollama for LLM synthesis. Frontend is an offline-capable PWA.

## Auth and access

- Login with username + password (argon2 hash). Session via JWT in an httpOnly, Secure, SameSite=Strict cookie.
- Every route requires auth except /login and /signup (invite code). Rate limit login attempts.
- Must run behind HTTPS (reverse proxy) since passwords cross the wire.
- Admin can deactivate a user (active = 0), which kills their access immediately.
- Public repo warning: .env, the .db file, join codes, and real scouting data must be gitignored.

## Schema (SQLite)

### Core: users and settings

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE,
  display_name TEXT,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('scout','strategist','admin')),
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE invite_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  code_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'scout' CHECK (role IN ('scout','strategist','admin')),
  expires_at TEXT,
  used_by INTEGER REFERENCES users(id),
  created_by INTEGER REFERENCES users(id),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE app_settings (
  key TEXT PRIMARY KEY,   -- team_number, active_event_key, tba_api_key_ref, ollama_model
  value TEXT NOT NULL
);

CREATE TABLE audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER REFERENCES users(id),
  action TEXT NOT NULL,
  detail TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### Event data (mostly auto-imported from TBA)

```sql
CREATE TABLE seasons (
  year INTEGER PRIMARY KEY,
  game_name TEXT,
  scouting_schema TEXT NOT NULL,  -- JSON: form fields, types, scoring weights
  prescout_schema TEXT            -- JSON: pre-scout form fields (can differ from live)
);

CREATE TABLE events (
  key TEXT PRIMARY KEY,           -- TBA key e.g. 2026wasno
  season INTEGER NOT NULL REFERENCES seasons(year),
  name TEXT, start_date TEXT, end_date TEXT
);

CREATE TABLE frc_teams (
  number INTEGER PRIMARY KEY,
  name TEXT, city TEXT, state_prov TEXT
);

CREATE TABLE matches (
  key TEXT PRIMARY KEY,           -- TBA match key
  event_key TEXT NOT NULL REFERENCES events(key),
  comp_level TEXT, set_number INTEGER, match_number INTEGER,
  red_score INTEGER, blue_score INTEGER,
  played INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE match_teams (
  match_key TEXT NOT NULL REFERENCES matches(key),
  team_number INTEGER NOT NULL REFERENCES frc_teams(number),
  alliance TEXT NOT NULL CHECK (alliance IN ('red','blue')),
  slot INTEGER,
  PRIMARY KEY (match_key, team_number)
);

CREATE TABLE external_stats (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL CHECK (source IN ('tba','statbotics')),
  team_number INTEGER, event_key TEXT,
  data TEXT NOT NULL,             -- JSON snapshot
  fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### Live scouting (event-day data)

```sql
CREATE TABLE scouting_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_key TEXT NOT NULL REFERENCES matches(key),
  team_number INTEGER NOT NULL REFERENCES frc_teams(number),
  scouter_id INTEGER NOT NULL REFERENCES users(id),
  data TEXT NOT NULL,             -- JSON validated against seasons.scouting_schema
  notes TEXT,
  status TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok','flagged','rejected')),
  client_uuid TEXT UNIQUE,        -- dedupe for offline resubmits
  submitted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE pit_scouting (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_key TEXT NOT NULL REFERENCES events(key),
  team_number INTEGER NOT NULL REFERENCES frc_teams(number),
  scouter_id INTEGER NOT NULL REFERENCES users(id),
  data TEXT NOT NULL,
  submitted_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE pit_photos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pit_id INTEGER NOT NULL REFERENCES pit_scouting(id) ON DELETE CASCADE,
  path TEXT NOT NULL
);

CREATE INDEX idx_scout_team ON scouting_entries(team_number);
CREATE INDEX idx_scout_match ON scouting_entries(match_key);
CREATE INDEX idx_match_teams_team ON match_teams(team_number);
```

### Pre-scouting (SEPARATE AREA)

```sql
-- A batch = one prep effort for an upcoming event (a CSV import, a video session, etc.)
CREATE TABLE prescout_batches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  target_event_key TEXT REFERENCES events(key),  -- event we are prepping for
  name TEXT NOT NULL,
  source_type TEXT NOT NULL CHECK (source_type IN ('csv','video','manual','api')),
  source_file TEXT,
  created_by INTEGER REFERENCES users(id),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE prescout_entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL REFERENCES prescout_batches(id) ON DELETE CASCADE,
  team_number INTEGER NOT NULL REFERENCES frc_teams(number),
  source_event_key TEXT REFERENCES events(key),  -- where the team was observed (past event)
  source_match_key TEXT REFERENCES matches(key),
  data TEXT NOT NULL,             -- JSON validated against seasons.prescout_schema
  video_url TEXT,
  notes TEXT,
  confidence INTEGER CHECK (confidence BETWEEN 1 AND 5),
  scouter_id INTEGER REFERENCES users(id),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_prescout_team ON prescout_entries(team_number);
CREATE INDEX idx_prescout_batch ON prescout_entries(batch_id);
```

### Derived data, AI, strategy

```sql
CREATE TABLE video_tracks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  match_key TEXT REFERENCES matches(key),
  team_number INTEGER,
  data TEXT NOT NULL,             -- frc_vision output (positions, score attribution)
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE ai_reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_key TEXT, team_number INTEGER, match_key TEXT,
  kind TEXT NOT NULL,             -- team_summary, match_strategy, picklist_rationale
  model TEXT,                     -- Ollama model used
  content TEXT NOT NULL,
  inputs_hash TEXT,               -- detect stale reports when data changes
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE picklists (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_key TEXT NOT NULL REFERENCES events(key),
  name TEXT NOT NULL,
  created_by INTEGER REFERENCES users(id),
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE picklist_items (
  picklist_id INTEGER NOT NULL REFERENCES picklists(id) ON DELETE CASCADE,
  team_number INTEGER NOT NULL REFERENCES frc_teams(number),
  rank INTEGER NOT NULL,
  tier TEXT, note TEXT,
  PRIMARY KEY (picklist_id, team_number)
);
```

## Pre-scouting area: how it should behave

- Own section of the app (own route group, own nav entry). Scouts can add entries; strategists and admins see everything.
- Three input paths: CSV import (your existing pre-scout CSVs), manual form, and video-based entries (link to match video + notes).
- Entries always reference a team and optionally the past event/match they were observed in.
- Model blending: final team score = live data weight + pre-scout weight + TBA/Statbotics weight. Pre-scout weight should fade as live matches accumulate for that team. Weights go in app_settings so they're tunable without code changes.
- Never write pre-scout rows into scouting_entries.

## Claude Code conventions

- Migrations via Alembic with render_as_batch=True (SQLite).
- Enable foreign_keys and WAL on every connection.
- Validate every JSON data payload against the season schema before insert. Reject unknown fields.
- Wrap offline sync batches in one transaction and dedupe on client_uuid.
- Nightly backup with sqlite3 .backup to a separate location. Keep the last 14.
- .env, *.db, uploads/, and any CSVs with real data go in .gitignore.
- Tests for auth (role gating on every route), schema validation, and the TBA importer.

## Feature roadmap

Tracked in the database below.

[Feature Roadmap](https://app.notion.com/p/553bd5ee2d4c43b297253cda36239507?pvs=21)