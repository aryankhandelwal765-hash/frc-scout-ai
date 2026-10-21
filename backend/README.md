# FRC Scout AI backend (Phase 0)

FastAPI + SQLAlchemy 2 + Alembic (SQLite, WAL). Source of truth: `docs/BUILD_SPEC.md`.
The frontend builds ONLY against `openapi.json` (regenerate with the export script below).

## Run

```bash
cd backend
python3.13 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # then edit; .env is gitignored. Dev: COOKIE_SECURE=false
.venv/bin/alembic upgrade head  # or: .venv/bin/python -m app.cli migrate
.venv/bin/python -m app.cli seed-season          # EXAMPLE season 2026 (placeholder schema)
.venv/bin/python -m app.cli create-admin         # prompts via getpass
#   scripted: ADMIN_PASSWORD=... .venv/bin/python -m app.cli create-admin --username NAME
.venv/bin/python -m uvicorn app.main:app_factory --factory --port 8000
.venv/bin/python -m app.cli import-event 2026xxxx   # or POST /admin/import/event/{key}
.venv/bin/python -m pytest
.venv/bin/python -m scripts.export_openapi       # rewrites openapi.json
```

## Config (env / backend/.env)

| var | notes |
|-----|-------|
| `ENV` | `dev` or `prod` |
| `SECRET_KEY` | JWT signing. **prod refuses to start** if missing/weak (<32 chars, low entropy, or common words). dev: random ephemeral key if unset |
| `COOKIE_SECURE` | default `true`; `false` only allowed when `ENV=dev` |
| `DATABASE_URL` | default `sqlite:///backend/scout.db` (gitignored) |
| `TBA_API_KEY` (alias `TBA_AUTH_KEY`) | new name wins |
| `OLLAMA_URL` (alias `OLLAMA_HOST`) | not used yet in Phase 0 |
| `CORS_ORIGINS` | comma list, never `*`. dev default `http://localhost:5173` |

## Deviations from the spec / decisions

- `invite_codes.code_prefix TEXT UNIQUE` added. Codes are stored argon2-hashed (salted), so they
  cannot be looked up by hash; the first 8 chars of the code locate the row, then argon2 verifies
  the full code.
- Admin passes every role gate (`require_role("strategist")` also admits admins).
- No `/docs`, `/redoc` or live `/openapi.json` (disabled so every served route is role-gated).
  Use the committed `openapi.json`.
- Usernames are stored lowercase (case-insensitive login).
- Password policy: 10-128 chars. Login limiter: 5 failures / 15 min per (username, IP), plus
  5 bad invite codes / 15 min per IP for signup. In-process, single worker only; behind a reverse
  proxy the "IP" is the proxy unless uvicorn `--proxy-headers` is configured.
- Importing an event for an unseeded season returns 409 (does not auto-create a season).
- `active`/`played` stay INTEGER 0/1 as in the spec (API exposes `active` as bool).
- Pre-scout: `app/services/prescout.py` writes only `prescout_*` tables (tested). No pre-scout
  HTTP routes in Phase 0.
- Alembic runs with `render_as_batch=True`. Later migrations that rebuild tables on SQLite should
  disable FK enforcement for that migration (batch recreate + `foreign_keys=ON` can fail).
- Routes: scouting entries (single + offline batch, deduped on `client_uuid`), pit scouting, picklists,
  pre-scout batches/entries, vision import (`POST /admin/import/vision/{event}` loads `VISION_SCOUT_TABLE`),
  team summary. Backups: `python -m app.cli backup` (sqlite backup API, keeps 14; schedule it from cron).
- Not yet implemented: pit photo upload, AI reports, the nightly backup scheduler itself.

## Cookies / CORS (frontend)

- Cookie `scout_session`: httpOnly, SameSite=Strict, Secure (unless dev + `COOKIE_SECURE=false`).
- Use same-origin requests via the Vite dev proxy (`/api` -> `http://localhost:8000`, strip the
  `/api` prefix; routes here have no prefix). SameSite=Strict needs same-site, so do not call
  :8000 directly from :5173 with fetch credentials unless CORS_ORIGINS includes it (it does in
  dev; localhost ports are same-site, so Strict still works).
- Always send `credentials: "include"` (or same-origin). Handle 401 (session over/deactivated),
  403 (role), 429 (`Retry-After` header).
