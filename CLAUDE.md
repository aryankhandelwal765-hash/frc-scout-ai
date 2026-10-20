# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

Two sub-projects, now fused into one system via `scout/engine/vision_loader.py`:

```
frc-scout-ai/
  strategy/       — FRC Scout AI: FastAPI + React + LLM synthesis + SQLite cache
  vision/  — CV pipeline: ByteTrack/YOLO robots + ball scoring + match renders
```

**Fusion point:** `strategy/scout/engine/vision_loader.py` reads
`vision/match_pipeline/results/*/estimate.json` + `identity.json`, joins by team
number, and injects vision-measured scoring into `build_strategy_brief()`. The
LLM now cites CV-measured ball counts alongside TBA OPR aggregates.

---

## strategy — FRC Scout AI

### Running the stack (dev)

```bash
cd strategy && ./start.sh           # Ollama + API + Vite together
# Or individually:
ollama serve
.venv/bin/python -m uvicorn scout.api.main:app --port 8000   # NO --reload — restart to reload
npm --prefix frontend run dev                                     # port 5173
```

App at http://localhost:5173 · API docs at http://localhost:8000/docs · Logs in `strategy/.logs/`.

**Environment gotcha:** the `uvicorn` console script is not always generated in
`.venv/bin/`. Always invoke it as `.venv/bin/python -m uvicorn ...`; `start.sh`
and the Dockerfile do this too. Logs in `.logs/api.log`.

### Running the stack (Docker / prod)

See `docs/deploy.md`. Three-container compose: `ollama`, `api` (Dockerfile builds
frontend + API into one image), `frontend` (nginx serving static + `/api` proxy).

```bash
cp .env.example .env             # fill TBA_AUTH_KEY, CORS_ORIGINS
npm --prefix frontend run build  # produces frontend/dist/
docker compose up -d --build
docker compose exec ollama ollama pull qwen3:8b
```

The API also serves `frontend/dist/` as static files, so a single-container
deploy works too (`docker run -p 8000:8000 frc-scout-ai`).

### Python environment

- Venv: `strategy/.venv` (Python 3.13). Always use `.venv/bin/python`.
- Providers are lazy-imported (only the active one loads), so the app runs on
  TBA without `statbotics` installed. FastAPI **requires** the venv.
- One-off scripts: `PYTHONPATH=<strategy root> .venv/bin/python <script>`.

### Frontend

```bash
npm --prefix frontend run dev     # dev server (Vite proxies /api → :8000)
npm --prefix frontend run build   # production build → frontend/dist/
npm --prefix frontend run lint    # oxlint
```

React 19 + TypeScript + `motion` (Apple-style springs). No router — page state
in `App.tsx`. Two pages: `MatchAnalysisPage` (dashboard + parallel Strategy +
Captain reads + BotSnapshot + ScoutingVideo) and `CaptainPage` (single briefing).

### TBA API key

Required for live data. Set `TBA_AUTH_KEY` in `.env`, which is loaded into
the environment. Without it, all stats return None and the UI says
"No TBA API key configured".

### Architecture — strategy

```
scout/
  config.py             — paths, model names (SYNTHESIS_MODEL=qwen3:8b),
                          DATA_PROVIDER, TBA key loader
  db.py                 — SQLite cache layer (kv_cache table); init on API startup;
                          TTLs: TBA 24h · predictions 1h · synthesis 12h
  settings.py           — my_team persistence (JSON at scout/user_settings.json)
  api/main.py           — FastAPI app; NDJSON streaming; StaticFiles mount for
                          single-container deploy; CORS_ORIGINS env-driven
  engine/
    data_source.py      — dispatcher: "tba" | "statbotics"
    tba_loader.py       — TBA v3 client; 2-tier cache (10min in-mem → 24h SQLite → TBA);
                          _phase_points ZEROES endgame (2026 season is teleop-only)
    statbotics_loader.py — Statbotics v3 (lazy import; often down)
    prediction_engine.py — EPA win-prob + ML score model; predict_match() = main entry
    archetype_engine.py  — playstyle + phase-share classification
    volatility_engine.py — floor/ceiling/consistency from epa_sd & epa_max
    chemistry_engine.py  — alliance synergy from archetypes
    vision_loader.py     — reads vision output; get_team_vision aggregates
                           per-team stats across an event; build_scouting_notes()
                           auto-generates the notes textarea content
    synthesis_engine.py  — GOD NODE: build_strategy_brief() merges quant + vision +
                           notes → LLM prompt; captain/coach/bot variants
    ollama_client.py     — Ollama HTTP client; yields ("thinking"|"content", text) events
  training/              — dataset builders + model trainers (offline)

frontend/src/
  types.ts               — shared TS types
  api.ts                 — fetch + ReadableStream NDJSON helpers
  useReader.ts           — single streaming read hook (strategy/coach/captain)
  useBots.ts             — multi-bot streaming hook
  Dashboard.css          — glass/backdrop-filter design tokens
  components/
    Dashboard.tsx        — VictoryBanner + AllianceSection ×2 + RobotCard grid
    VictoryBanner.tsx    — split-gradient win-prob banner
    AllianceSection.tsx  — per-side glass card (OPR, chemistry, phase bars)
    RobotCard.tsx        — compact bot card w/ tap-to-expand (motion springs)
    BotSnapshot.tsx      — 6-bot 30s clip grid (ffmpeg-cut from render mp4)
    ScoutingVideo.tsx    — video picker + render-on-demand button
    MatchSetup.tsx       — team inputs + TBA match import + "Auto-fill from Vision"
    Sidebar / StatusBar / SynthCard / StreamRead / BotAnalysis — UI plumbing
  pages/
    MatchAnalysisPage.tsx — dashboard + PARALLEL Strategy+Captain reads; hides
                            MatchSetup post-analyze; myTeam determines "our side"
                            (NOT the setup's default 'red')
    CaptainPage.tsx      — dedicated captain briefing (single alliance)
  App.tsx                — sidebar + page state
```

### Key data-flow

```
predict_match()  ─┐
                  ├─► build_strategy_brief()  ── injects vision-measured scoring
get_team_vision() ┘         │                    when available
                            ▼
                    SYSTEM_PROMPT + text
                            │
                            ▼
                    stream_chat_verbose()  ── yields ("thinking"|"content", text)
                            │
                            ▼
                    NDJSON events over HTTP
                            │
                            ▼
                    useReader / useBots (React) ── renders progress bar + markdown
```

### Ollama models

| purpose | model | typical latency |
|---------|-------|-----------------|
| Strategy / captain / coach synthesis | `qwen3:8b` | ~5–10s per read |
| Per-robot bot-by-bot | `qwen3:8b` (`think=False`) | ~3–5s per bot |
| Role assignment (all 6 bots one shot) | `qwen3:8b` (`fmt="json"`) | ~4–5s |

Was `qwen3.6:35b-a3b` (~70s per read) — swapped for the 8b for 10× speedup with
minor quality loss. Ollama stops when the Mac sleeps; restart `ollama serve`.

### 2026 game rules baked in

**Endgame/climb is worth ~zero this season.** `_phase_points` in `tba_loader.py`
hard-zeroes the endgame return value. All prompts (SYSTEM_PROMPT,
BOT_SYSTEM_PROMPT, ROLES_SYSTEM_PROMPT, coach/captain prompts) explicitly
forbid mentioning endgame. Frontend robot cards and phase bars show A/T only.

### Cache layer

- SQLite at `scout/cache.db` (env: `CACHE_DB_PATH`; `/data/cache.db` in Docker)
- Kicks in on `init_db()` at API startup
- `GET /cache/stats` for diagnostic (entries, size, oldest age)
- `DELETE /cache` to wipe (do this after prompt changes so old syntheses aren't served)
- Measured impact: first `GET /team/254` = 6.3s, subsequent = 16ms (**390×**)

### Key API endpoints

```
GET  /health                          — status + Ollama + TBA reachability
GET  /cache/stats · DELETE /cache
GET  /settings · POST /settings       — my_team persistence
GET  /events/{key}/matches            — TBA match schedule (SQLite-cached)
GET  /team/{team}?year=2026
POST /predict                         — quant only (fast)
POST /roles                           — one-shot per-robot job phrases (JSON mode)
POST /analyze/stream                  — Strategy read (NDJSON stream)
POST /coach/stream                    — My-team coaching (NDJSON stream)
POST /captain/stream                  — Captain briefing (NDJSON stream)
POST /bots/stream                     — Per-robot reads (NDJSON stream)

GET  /vision/team/{team}              — aggregated CV stats across event
GET  /vision/match/{key}              — per-match vision output (all 6 robots)
POST /vision/notes                    — auto-generated scouting notes text
GET  /vision/snapshot/{team}          — pre-match snapshot + clip URL

GET  /scouting/matches                — every match w/ preproc + render status
GET  /scouting/videos                 — list rendered mp4s
GET  /scouting/videos/{key}           — serve rendered mp4
POST /scouting/render/{key}           — QUEUE a render, returns 202 immediately
GET  /scouting/render/{key}           — render job status (idle/running/done/error)
GET  /scouting/clip/{team}/{key}      — 30s ffmpeg-cut highlight (~1 MB, cached)
```

### Tests

```bash
strategy/.venv/bin/python -m pip install -r requirements-dev.txt
strategy/.venv/bin/python -m pytest          # 115 tests, no network
strategy/.venv/bin/python -m pytest -m "not slow"   # skip the video e2e
```

Root `conftest.py` puts `strategy/`, `vision/classic/` and
`vision/match_pipeline/` on `sys.path`. An autouse fixture blocks real sockets,
so an unmocked TBA/Statbotics/Ollama call fails loudly. Detector tests skip
themselves when YOLO weights are absent.

### Known issues / TODO

- Win probability may still peg high on some matchups — OPR scale (~300/team)
  vs. tuned `win_probability_from_diff` sigma=100. Retune sigma to ~250–400 or
  normalize OPR → EPA scale in `feature_engineering.py`.
- `models/*.pkl` trained on sklearn 1.6.1, venv has 1.8.0 →
  `InconsistentVersionWarning` on load (non-blocking).
- The render job registry is in-process (a single-worker ThreadPoolExecutor in
  `api/main.py`). It no longer blocks the request thread, but state is lost on
  API restart — a render running at restart just reads as missing again. Move to
  RQ/Celery if this ever hits real traffic.
- `examples/2026hop/path_quality.csv` does not reproduce the illegal-crossing
  rates quoted in `vision/match_pipeline/README.md`; it is probably stale. Rerun
  `validate_paths.py` against the current field map.

---

## vision — CV pipeline

### Setup

```bash
cd vision
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg tesseract        # non-pip prereqs
pip install yt-dlp                   # match video downloader
```

### Two workflows

**1. Standalone scoring on any clip** — from `vision/classic/track_match.py`:
```bash
python classic/track_match.py --input clip.mp4 --zones zones.json --outdir out --ball-color yellow
```
Outputs per-robot ball tally + annotated mp4. See `vision/README.md` for
the full flow (goal-zone picker, tuning levers, HSV calibration).

**2. TBA batch pipeline** — from `vision/match_pipeline/`. This is what feeds
strategy's vision fusion. Multi-stage:

```bash
cd vision/match_pipeline
../venv/bin/python download_matches.py           # yt-dlp all matches for an event
../venv/bin/python batch_run.py                  # scoring estimator per video → estimate.json
../venv/bin/python batch_trajectories.py         # ByteTrack per video → tracks.json + trajectory.csv
../venv/bin/python batch_auto.py                 # auto-start detection → auto_paths.json
../venv/bin/python render_full_match.py --match 2026hop_qm28   # per-match render
```

Each `results/<match_key>/` gets: `tracks.json`, `identity.json`,
`auto_paths.json`, `estimate.json`, `board_series.csv`, `trajectory.csv`,
`full_match_render.mp4`.

**Preprocessing = tracks + identity + auto_paths** must all exist before
`render_full_match.py` will work. The strategy API's `/scouting/matches` endpoint
reports per-match render status (`fresh` / `stale` / `missing` / no-preproc).
Stale = the sha256 of `render_full_match.py` + `stitch.py` + `topdown.py` no
longer matches the `.render_scripts.sha256` stamp written beside the mp4. (It
used to compare mtimes, which went stale on any checkout.) Trigger a re-render
from the Scouting Video panel; it queues on the API's worker pool and the UI
polls `GET /scouting/render/{key}`.

### Architecture — vision (root level)

```
classic/            — v1 classic-CV tracker, self-contained:
  track_match.py      — standalone entry (robots + balls + scoring)
  config.py           — HSV color presets, tunable params
  detection.py        — color masks, robot + ball blob extraction
  tracking.py         — nearest-centroid tracker w/ grace period
  scoring.py          — possession attribution + goal-zone scoring
  overlay.py          — annotated-video drawing
  scoring_estimator.py — rate-apportionment estimator (match_pipeline imports it)
  make_test_clip.py   — synthetic clip w/ a known answer (RED 2 / BLUE 1)
  calibrate.py / tune_color_masks.py — HSV tuning helpers
train_detector.py   — fine-tune YOLO on labeled frames
annotate.py / extract_frames.py — labeling toolchain
prelabel.py         — drafts YOLO labels across many matches (fine-tuned detector, person veto,
                      nested-box merge, colour-by-elimination); writes dataset_v2/ + review_order.txt
review/             — AI box review: a vision model says keep/delete/recolour per prelabel box,
                      scored against human labels (ai_review*.py, review_bench.py; see its README)
match_pipeline/robot_filters.py — pure-python person veto / nested merge / alliance elimination
```

### Architecture — vision/match_pipeline/

```
download_matches.py      — yt-dlp per TBA event
batch_run.py             — scoring estimator (parallel workers)
batch_trajectories.py    — ByteTrack (parallel workers)
batch_auto.py            — auto path detection
identify.py              — bumper OCR → team ID w/ voting
tba.py                   — TBA event/match/OPR fetchers (separate from strategy's)
stitch.py                — combines broadcast + top-down + HUD into one frame
topdown.py + topdown_paths.py — field homography + trajectory rendering
auto_paths.py            — per-robot auto trajectory extraction
render_full_match.py     — final MATCH render (used by strategy API on demand)
```

### Vision fusion — how strategy reads it

`strategy/scout/engine/vision_loader.py::_load_match()` reads
`results/<match_key>/estimate.json` (per-robot `est_balls_scored`, `activity`,
`position`) and `identity.json` (track_id → team w/ confidence). It approximates
slot → team by pairing estimate slots to identity tracks sorted by confidence
descending. `get_team_vision(team)` aggregates across every match in an event
prefix (default `"2026hop"`). Cached 6h in the same SQLite as TBA data.

### Deploy considerations

- **vision videos are huge** (~50–250 MB each) — don't commit. `.gitignore`
  in strategy already excludes them; do the same at the root.
- **Model weights** (`.pt` files) also huge — Git LFS or off-repo storage.
- The `render_full_match.py` script has to be reachable from the strategy API
  container. In Docker, either mount `vision/` as a volume or bake it into
  the API image (increases image size).
