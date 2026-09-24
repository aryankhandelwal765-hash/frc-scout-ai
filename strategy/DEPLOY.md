# FRC Scout AI — Deployment Guide

## Quick start (Docker)

```bash
# 1. Configure
cp .env.example .env
# Edit .env: set TBA_AUTH_KEY, adjust CORS_ORIGINS for your domain

# 2. Build the frontend (once, or on any UI change)
npm --prefix frontend install
npm --prefix frontend run build

# 3. Bring up the stack
docker compose up -d --build

# 4. Pull Ollama models (first time only, ~5 GB per model)
docker compose exec ollama ollama pull qwen3:8b       # synthesis + bots
docker compose exec ollama ollama pull qwen2.5:3b     # fast/coding

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

## Environment variables (.env)

| Var | Default | Notes |
|-----|---------|-------|
| `TBA_AUTH_KEY` | (empty) | **Required** for live TBA data |
| `CORS_ORIGINS` | `*` | Comma-separated. Lock down in prod: `https://scout.example.com` |
| `OLLAMA_BASE` | `http://localhost:11434` | Auto-set to `http://ollama:11434` in compose |
| `CACHE_DB_PATH` | `./scout/cache.db` | Auto-set to `/data/cache.db` in compose |

## Production checklist

- [ ] `.env` populated with real `TBA_AUTH_KEY`
- [ ] `CORS_ORIGINS` locked to your actual frontend domain(s)
- [ ] Reverse proxy in front (TLS/HTTPS termination): Caddy, Traefik, nginx
- [ ] Ollama models pulled: `qwen3:8b` (required), `qwen2.5:3b` (optional)
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
