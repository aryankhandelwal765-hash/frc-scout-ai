# FRC Scout AI

Turns FRC match video and The Blue Alliance data into an alliance-strategy briefing — computer vision measures what each robot actually did, a quantitative model predicts the match, and a local LLM writes the scouting read.

## The problem

FRC scouting is manual. Someone sits in the stands with a clipboard (or a tablet form) and tallies what each robot does, match after match. The data is thin, inconsistent between scouts, and by the time you need it — standing at the alliance selection table, or three minutes before a match — nobody has time to read a spreadsheet.

This replaces the clipboard with two things: a CV pipeline that measures per-robot scoring straight from match video, and a strategy layer that fuses those measurements with TBA's aggregate stats (OPR/EPA) and streams a plain-English briefing.

## Repo layout

Two sub-systems in one repo, joined at a single seam.

```
frc-scout-ai/
  vision/                 the measurement half — what each robot actually did
    track_match.py        classic-CV entry point (no training, CPU-only)
    detector_estimator.py YOLO-based per-robot scoring estimator
    match_pipeline/       batch jobs over downloaded match video
      batch_run.py          run the estimator across every video
      identify.py           team identity by bumper-number OCR voting
      export_to_strategy.py hand measured scouting data to strategy/
  strategy/               the decision half — what to do about it
    scout/                the Python package
      engine/               prediction, archetypes, volatility, chemistry,
                            vision ingest, LLM synthesis
      api/                  FastAPI app, NDJSON streaming endpoints
      training/             offline dataset builders + model trainers
      pages/                Streamlit pages (legacy but functional)
    frontend/             React 19 + TypeScript dashboard
```

## Architecture

```
vision/                      strategy/
 match video                      TBA API (OPR/EPA, schedules)
      │                                │
      ▼                                ▼
 YOLO robot detect ──┐          prediction_engine
 + bumper-colour     │          archetype / volatility / chemistry engines
   alliance class    │                 │
      ▼              │                 │
 per-robot tracking  │                 │
 + scoreboard OCR    │                 │
      ▼              │                 │
 bumper-number       │                 │
 identity voting     │                 │
      ▼              │                 │
 results/<match>/    │                 │
   estimate.json  ───┴──► vision_loader.py ──► synthesis_engine.build_strategy_brief()
   identity.json                                        │
                                                        ▼
                                              Ollama (qwen3:8b), streamed
                                                        │
                                              NDJSON over HTTP
                                                        ▼
                                              React dashboard
```

**The fusion point is `strategy/scout/engine/vision_loader.py`.** It reads `vision/match_pipeline/results/*/estimate.json` and `identity.json`, joins them to TBA data by team number, and injects CV-measured ball counts into the LLM prompt. The briefing cites measured scoring alongside TBA aggregates rather than guessing from OPR alone.

### `vision` — the CV pipeline

Per match video, run as a batch job (`match_pipeline/batch_run.py`, one process per video):

1. **Detect** robots with a YOLO detector trained on bumper colour, so alliance comes from the bumper class rather than a field-half split.
2. **Track** each robot across frames into persistent per-alliance IDs.
3. **Measure** per-robot scoring activity from ball motion around each robot's box.
4. **OCR the scoreboard** for authoritative per-alliance totals, then apportion each alliance's real total across its robots by measured activity. Alliance totals are exact; the per-robot split is an estimate.
5. **Identify** teams by voting on OCR'd bumper numbers (`identify.py`), scored only against the six teams TBA says are in the match — so partial reads still resolve, and a ~16%-per-frame read rate becomes a confident per-track identity.

`export_to_strategy.py` writes the result into the per-team-per-match CSV schema strategy's aggregator already expected from hand scouting.

There is also a standalone classic-CV path (`track_match.py`: HSV thresholding + contour detection + nearest-centroid tracking, no training, CPU-only) kept for the zero-setup case — `make_test_clip.py` generates a synthetic clip with a known answer to validate the pipeline end to end.

### `strategy` — the strategy system

- **`engine/tba_loader.py`** — TBA v3 client behind a two-tier cache (10-min in-memory → 24h SQLite → network). Measured: first `GET /team/254` 6.3s, subsequent 16ms.
- **`engine/prediction_engine.py`** — EPA win probability plus an ML score model; `predict_match()` is the entry point.
- **`engine/archetype_engine.py` / `volatility_engine.py` / `chemistry_engine.py`** — playstyle classification, floor/ceiling/consistency from EPA spread, and alliance synergy from archetype pairings.
- **`engine/synthesis_engine.py`** — the god node. Merges quantitative output, vision measurements and free-text notes into the LLM prompt. Four variants: strategy read, captain's briefing, my-team coaching, per-robot breakdown.
- **`engine/ollama_client.py`** — streams `("thinking" | "content", text)` events, so the UI can show a real progress bar during the model's reasoning phase instead of hanging.
- **`api/main.py`** — FastAPI, NDJSON streaming endpoints, serves the built frontend as static files for single-container deploys.

The React frontend (no router — page state in `App.tsx`) renders a win-probability banner, per-robot bento cards, and live streaming progress with token counts and ETA.

### Notable design decisions

- **Endgame is hard-zeroed.** `_phase_points` in `tba_loader.py` returns zero for endgame, and every prompt forbids mentioning it, because climb is worth ~nothing in the 2026 game. Phase bars show auto/teleop only.
- **Unmeasured ≠ zero.** strategy's aggregator reads a missing column as an empty series and returns 0, which reads as "this team never does that" rather than "we didn't measure it." `export_to_strategy.py` guards this explicitly — it's a real way to mislead a pick list.
- **Streaming state updaters must be pure.** React 18 StrictMode double-invokes them; an in-place mutation renders every token twice.

## Tech stack

**Vision** — Python, YOLOv8 (Ultralytics), ByteTrack, OpenCV, Tesseract OCR, NumPy
**Backend** — FastAPI, Uvicorn, SQLite, pandas, scikit-learn
**LLM** — Ollama running `qwen3:8b` locally
**Frontend** — React 19, TypeScript, Vite, `motion`
**Data** — The Blue Alliance API v3, Statbotics v3 (fallback)
**Deploy** — Docker Compose (ollama + api + nginx frontend)

## Setup

Requires Python 3.13, Node 18+, [Ollama](https://ollama.com), and a free [TBA read API key](https://www.thebluealliance.com/account).

```bash
cp .env.example .env        # fill in TBA_AUTH_KEY
```

### Strategy app

```bash
cd strategy
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
npm --prefix frontend install
ollama pull qwen3:8b

./start.sh                  # Ollama + API + Vite together
```

App at http://localhost:5173, API docs at http://localhost:8000/docs.

Running the pieces individually:

```bash
ollama serve
.venv/bin/python -m uvicorn scout.api.main:app --port 8000   # no --reload
npm --prefix frontend run dev
```

> The `uvicorn` console script in `.venv/bin/` is sometimes missing — use `.venv/bin/python -m uvicorn` instead. `start.sh` calls the script directly and fails silently if it's absent; check `strategy/.logs/api.log`.

### Vision pipeline

```bash
cd vision
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Sanity-check with no footage: synthetic clip, known answer (RED 2 / BLUE 1)
python make_test_clip.py
python track_match.py --input test_clip.mp4 --zones test_zones.json --outdir out
```

Batch pipeline over real matches:

```bash
cd match_pipeline
python download_matches.py --event 2026hop   # needs TBA_AUTH_KEY
python batch_run.py --workers 12             # → results/<match>/estimate.json
python identify.py                           # → results/<match>/identity.json
python export_to_strategy.py                       # → strategy's scouting CSV
```

Model weights and match video are gitignored — weights download on first use via Ultralytics, and match video comes from `download_matches.py`. Pick a static broadcast angle; the tracker has no camera-motion compensation.

### Docker

```bash
cp .env.example .env
npm --prefix strategy/frontend run build
cd strategy && docker compose up -d --build
docker compose exec ollama ollama pull qwen3:8b
```

See `strategy/DEPLOY.md` for the full deployment checklist.

## Screenshots

<!-- TODO: add a screenshot of the Match Analysis dashboard (win banner + robot bento cards) -->
<!-- TODO: add a GIF of a streaming strategy read, showing the reasoning → writing progress bar -->
<!-- TODO: add a still from an annotated match render showing tracked robots + per-robot ball tallies -->

## Status

Personal project, built around the 2026 FRC season. Game-specific assumptions (notably the zeroed endgame) are baked into the scoring logic and prompts.
