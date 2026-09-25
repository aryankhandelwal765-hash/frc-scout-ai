# FRC Scout AI — Deployment Guide

## Quick start (Docker)

All commands below run from `strategy/`.

```bash
# 1. Configure
cp ../.env.example .env
# Edit .env: set TBA_AUTH_KEY, adjust CORS_ORIGINS for your domain.
# Optionally set OLLAMA_MODEL; it defaults to qwen3:8b.

# 2. Build the frontend (once, or on any UI change)
npm --prefix frontend install
npm --prefix frontend run build

# 3. Bring up the stack
docker compose up -d --build

# 4. Pull the Ollama model (first time only, ~5 GB)
docker compose exec ollama ollama pull "${OLLAMA_MODEL:-qwen3:8b}"

# 5. Open http://localhost/
```

Three containers come up:
- `ollama` — local LLM (:11434)
- `api` — FastAPI + SQLite cache (:8000)
- `frontend` — nginx serving built React (:80, proxies /api to api container)

## Single-container deploy (no nginx)

The API also serves the built frontend from `frontend/dist/` — you can skip the
nginx container entirely and hit `http://<host>:8000/` for both UI and API.

```bash
npm --prefix frontend run build   # produces frontend/dist/
docker build -t frc-scout-ai .
docker run -p 8000:8000 --env-file .env -v frc-data:/data frc-scout-ai
```

## Cache layer

Persistent SQLite cache at `/data/cache.db` (env: `CACHE_DB_PATH`):

| Data | TTL | Reason |
|------|-----|--------|
| TBA team stats | 24 h | Slow, rate-limited API |
| TBA event matches | 24 h | Same |
| Predictions | 1 h | Quant is cheap but re-computing per click adds up |
| LLM synthesis | 12 h | ~5–10 s per read w/ qwen3:8b |

Measured speedup on `/team/254`: **6.3 s → 16 ms (390×)**.

Cache management:
```bash
curl http://localhost:8000/cache/stats            # entries + disk usage
curl -X DELETE http://localhost:8000/cache        # wipe all
```

The volume `api_data:/data` in docker-compose keeps the cache across rebuilds.

## Ollama model

`OLLAMA_MODEL` selects the model behind the strategy, captain, coach and
per-robot reads. It defaults to `qwen3:8b`, which is what the prompts are tuned
for at roughly 5-10 s per read.

```bash
# .env
OLLAMA_MODEL=qwen3:8b
```

The larger `qwen3.6:35b-a3b` is **optional**. It takes around 70 s per read for
a modest quality gain, so it is not the default. To use it, set `OLLAMA_MODEL`
and pull it before bringing the API up.

## Environment variables (.env)

| Var | Default | Notes |
|-----|---------|-------|
| `TBA_AUTH_KEY` | (empty) | **Required** for live TBA data |
| `CORS_ORIGINS` | `*` | Comma-separated. Lock down in prod: `https://scout.example.com` |
| `OLLAMA_HOST` | `http://localhost:11434` | Auto-set to `http://ollama:11434` in compose. `OLLAMA_BASE` is accepted as a legacy alias |
| `OLLAMA_MODEL` | `qwen3:8b` | Model behind every synthesis read. The larger `qwen3.6:35b-a3b` is optional and ~10x slower per read |
| `CACHE_DB_PATH` | `./scout/cache.db` | Auto-set to `/data/cache.db` in compose |

## Production checklist

- [ ] `.env` populated with real `TBA_AUTH_KEY`
- [ ] `CORS_ORIGINS` locked to your actual frontend domain(s)
- [ ] Reverse proxy in front (TLS/HTTPS termination): Caddy, Traefik, nginx
- [ ] Ollama model pulled: whatever `OLLAMA_MODEL` names (default `qwen3:8b`)
- [ ] Persistent volume mounted at `/data` for cache DB survival
- [ ] Health check endpoint monitored: `GET /health` → 200 w/ ollama_running=true
- [ ] Frontend rebuilt after any UI change: `npm --prefix frontend run build`
- [ ] For GPU: uncomment the `deploy.resources` block in docker-compose.yml
      and install `nvidia-container-toolkit` on the host

## Troubleshooting

**`ollama_running: false` in /health**
- Ollama container not up or unreachable. Check: `docker compose logs ollama`

**`provider_configured: false` in /health**
- `TBA_AUTH_KEY` missing from .env. Get a free read key at
  https://thebluealliance.com/account

**Slow first request per team**
- Expected. TBA fetch is ~5–10 s per team the first time. Cached afterward.

**Rendering endpoint (`POST /scouting/render/{key}`) fails**
- Requires the `vision` sub-project + its venv at
  `../vision/venv/bin/python`. Not needed for the core prediction stack.
