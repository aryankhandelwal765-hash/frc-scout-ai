"""FRC Scout AI — FastAPI backend.

Drives the quant + strategy-captain pipeline for the React frontend (and any other
client). Streaming endpoints emit NDJSON events (one JSON object per line) so the
frontend can render the reasoning-progress UX:

    {"type": "thinking", "text": "..."}   reasoning tokens (before the answer)
    {"type": "content",  "text": "..."}   answer tokens
    {"type": "bot_start"/"bot_done", "team": N, "name": "..."}  (bots stream)
    {"type": "done"}                       stream finished
    {"type": "error", "text": "..."}       failure

Run from the repo root:
    uvicorn scout.api.main:app --reload --port 8000
"""

import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Scouting video directory (vision output).
SCOUTING_ROOT = Path(__file__).resolve().parents[4] / "vision" / "match_pipeline" / "results"

from scout.config import BOT_MODEL, SYNTHESIS_MODEL
from scout.db import cache_clear, cache_purge_expired, cache_stats, init_db
from scout.engine import data_source, ollama_client
from scout.engine.archetype_engine import classify_archetype
from scout.engine.ollama_client import stream_chat_verbose
from scout.engine.prediction_engine import predict_match
from scout.engine.synthesis_engine import (
    SYSTEM_PROMPT,
    build_captain_brief,
    build_coach_brief,
    build_strategy_brief,
    generate_roles,
    stream_bot_analysis,
    synthesize,
)
from scout.engine.vision_loader import (
    build_scouting_notes,
    get_match_vision,
    get_team_vision,
)
from scout.engine.volatility_engine import profile_volatility
from scout.settings import get_my_team, set_my_team

app = FastAPI(
    title="FRC Scout AI",
    description="EPA/OPR quant model + Ollama strategy-captain synthesis for 2026 FRC.",
    version="2.1.0",
)

# CORS: locked down via CORS_ORIGINS env var (comma-separated) in production.
# Defaults to "*" for dev; set explicit origins before deploying.
_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    cache_purge_expired()  # keep DB lean on boot


# ── Request models ─────────────────────────────────────────────────────────────


class MatchupRequest(BaseModel):
    red_teams: list[int] = Field(..., description="Exactly 3 red team numbers")
    blue_teams: list[int] = Field(..., description="Exactly 3 blue team numbers")
    year: int = 2026
    notes: str | None = None
    model: str | None = None

    @field_validator("red_teams", "blue_teams")
    @classmethod
    def _exactly_three(cls, v: list[int]) -> list[int]:
        if len(v) != 3:
            raise ValueError("each alliance must have exactly 3 teams")
        return v


class CoachRequest(MatchupRequest):
    my_team: int | None = Field(None, description="Defaults to the saved team setting")


class CaptainRequest(MatchupRequest):
    our_side: str | None = Field(None, description="'red' or 'blue'; defaults from saved team")


class SettingsBody(BaseModel):
    my_team: int | None = None


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _event(kind: str, **extra) -> str:
    return json.dumps({"type": kind, **extra}) + "\n"


def _require_ollama():
    if not ollama_client.is_running():
        raise HTTPException(503, "Ollama is not running (start it with `ollama serve`).")


def _stream_read(system: str, user: str, model: str):
    """NDJSON generator for a single streamed read (thinking + content + done)."""

    def gen():
        try:
            for kind, text in stream_chat_verbose(model, system, user):
                yield _event(kind, text=text)
            yield _event("done")
        except Exception as e:
            yield _event("error", text=str(e))

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ── Status / settings ───────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "data_provider": data_source.provider_name(),
        "provider_configured": data_source.is_configured(),
        "provider_reachable": data_source.is_reachable(),
        "ollama_running": ollama_client.is_running(),
        "synthesis_model": SYNTHESIS_MODEL,
        "bot_model": BOT_MODEL,
        "available_models": ollama_client.list_models(),
    }


@app.get("/cache/stats")
def cache_diag() -> dict:
    """Cache metrics: entries, disk usage, oldest entry age."""
    return cache_stats()


@app.delete("/cache")
def cache_wipe() -> dict:
    """Wipe all cached entries. Returns rows deleted."""
    return {"deleted": cache_clear()}


@app.get("/settings")
def read_settings() -> dict:
    return {"my_team": get_my_team()}


@app.post("/settings")
def write_settings(body: SettingsBody) -> dict:
    set_my_team(body.my_team)
    return {"my_team": get_my_team()}


# ── Data ────────────────────────────────────────────────────────────────────────


@app.get("/events/{event_key}/matches")
def event_matches(event_key: str) -> dict:
    matches = data_source.get_event_matches(event_key)
    return {"event_key": event_key, "matches": matches, "count": len(matches)}


# ── Scouting videos (vision output) ────────────────────────────────────────


@app.get("/scouting/videos")
def scouting_videos() -> dict:
    """List available annotated match videos (with robot bounding boxes)."""
    if not SCOUTING_ROOT.exists():
        return {"videos": [], "root": str(SCOUTING_ROOT)}
    videos = []
    for match_dir in sorted(SCOUTING_ROOT.iterdir(), reverse=True):
        if not match_dir.is_dir():
            continue
        # Prefer full_match_render.mp4 (annotated); fall back to any .mp4 in the dir.
        annotated = match_dir / "full_match_render.mp4"
        if annotated.exists():
            videos.append(
                {
                    "match_key": match_dir.name,
                    "file": "full_match_render.mp4",
                    "size_mb": round(annotated.stat().st_size / 1e6, 1),
                }
            )
            continue
        for mp4 in match_dir.glob("*.mp4"):
            videos.append(
                {
                    "match_key": match_dir.name,
                    "file": mp4.name,
                    "size_mb": round(mp4.stat().st_size / 1e6, 1),
                }
            )
            break
    return {"videos": videos, "count": len(videos)}


@app.get("/scouting/videos/{match_key}")
def scouting_video(match_key: str):
    """Stream an annotated match video by match key (e.g. 2026hop_qm20)."""
    match_dir = SCOUTING_ROOT / match_key
    if not match_dir.exists() or not match_dir.is_dir():
        raise HTTPException(404, f"No scouting output for match {match_key}")
    annotated = match_dir / "full_match_render.mp4"
    if not annotated.exists():
        mp4s = list(match_dir.glob("*.mp4"))
        if not mp4s:
            raise HTTPException(404, f"No video file in {match_key}")
        annotated = mp4s[0]
    return FileResponse(annotated, media_type="video/mp4", filename=annotated.name)


# Render freshness is decided by the CONTENT of the scripts that produce a
# render, not their mtime. mtime changes on any checkout, rebase or touch, which
# marked every render stale after an unrelated `git pull`. The hash of the three
# scripts is written beside the mp4 when a render finishes and compared here.
_RENDER_SCRIPTS = ("render_full_match.py", "stitch.py", "topdown.py")
_RENDER_STAMP = ".render_scripts.sha256"


def _render_scripts_hash() -> str:
    """sha256 over the render scripts' bytes, or '' if none are readable."""
    h = hashlib.sha256()
    found = False
    for name in _RENDER_SCRIPTS:
        path = SCOUTING_ROOT.parent / name
        if not path.exists():
            continue
        h.update(name.encode())
        h.update(path.read_bytes())
        found = True
    return h.hexdigest() if found else ""


def _render_is_fresh(match_dir: Path, current_hash: str) -> bool:
    """True when the render was produced by the scripts as they are now."""
    if not current_hash:
        return True  # nothing to compare against; do not cry stale
    stamp = match_dir / _RENDER_STAMP
    if not stamp.exists():
        return False  # rendered before stamping existed — re-render to be sure
    try:
        return stamp.read_text().strip() == current_hash
    except OSError:
        return False


@app.get("/scouting/matches")
def scouting_matches() -> dict:
    """List every match with preprocessing done + render status (fresh/stale/missing)."""
    if not SCOUTING_ROOT.exists():
        return {"matches": [], "root": str(SCOUTING_ROOT)}

    current_hash = _render_scripts_hash()

    matches = []
    for match_dir in sorted(SCOUTING_ROOT.iterdir(), reverse=True):
        if not match_dir.is_dir():
            continue
        has_tracks = (match_dir / "tracks.json").exists()
        has_identity = (match_dir / "identity.json").exists()
        has_autopaths = (match_dir / "auto_paths.json").exists()
        ready_to_render = has_tracks and has_identity and has_autopaths

        render_file = match_dir / "full_match_render.mp4"
        render_status = "missing"
        size_mb = 0.0
        if render_file.exists():
            size_mb = round(render_file.stat().st_size / 1e6, 1)
            render_status = "fresh" if _render_is_fresh(match_dir, current_hash) else "stale"

        matches.append(
            {
                "match_key": match_dir.name,
                "ready_to_render": ready_to_render,
                "render_status": render_status,
                "size_mb": size_mb,
            }
        )
    return {"matches": matches, "count": len(matches)}


# Renders take 1-5 minutes. They run on a small worker pool so the request
# thread is never occupied; the frontend polls GET /scouting/render/{key} for
# status. State is in-process, so it resets when the API restarts — a render
# that was running at restart is simply reported as missing again.
_RENDER_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
_RENDER_JOBS: dict[str, dict] = {}
_RENDER_LOCK = threading.Lock()


def _run_render(match_key: str, venv_python: Path, script_dir: Path) -> None:
    """Worker body. Runs off the request thread; records its own outcome."""
    import subprocess

    def _finish(**fields) -> None:
        with _RENDER_LOCK:
            _RENDER_JOBS[match_key].update(finished_at=time.time(), **fields)

    try:
        result = subprocess.run(
            [str(venv_python), "render_full_match.py", "--match", match_key],
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        _finish(status="error", error=f"Render timed out after 10 min for {match_key}")
        return
    except Exception as exc:  # pragma: no cover - defensive
        _finish(status="error", error=str(exc))
        return

    if result.returncode != 0:
        _finish(status="error", error=f"Render failed: {result.stderr[-500:]}")
        return

    render_file = SCOUTING_ROOT / match_key / "full_match_render.mp4"
    if not render_file.exists():
        _finish(status="error", error="Render script ran but produced no output file")
        return

    # Stamp the render with the hash of the scripts that produced it, so
    # /scouting/matches can tell fresh from stale without consulting mtimes.
    try:
        (SCOUTING_ROOT / match_key / _RENDER_STAMP).write_text(_render_scripts_hash())
    except OSError:
        pass  # a missing stamp only means the render reads as stale

    _finish(
        status="done",
        size_mb=round(render_file.stat().st_size / 1e6, 1),
        output=result.stdout[-500:],
    )


@app.post("/scouting/render/{match_key}", status_code=202)
def scouting_render(match_key: str) -> dict:
    """Queue a render for a match. Returns immediately; poll GET for status."""
    match_dir = SCOUTING_ROOT / match_key
    if not match_dir.exists():
        raise HTTPException(404, f"No preprocessing for {match_key}")
    if not (match_dir / "tracks.json").exists():
        raise HTTPException(400, f"{match_key} missing tracks.json — run full pipeline first")

    script_dir = SCOUTING_ROOT.parent
    venv_python = script_dir.parent / "venv" / "bin" / "python"
    if not venv_python.exists():
        raise HTTPException(500, f"vision venv not found at {venv_python}")

    with _RENDER_LOCK:
        job = _RENDER_JOBS.get(match_key)
        if job and job["status"] == "running":
            return {"match_key": match_key, "status": "running", "started_at": job["started_at"]}
        _RENDER_JOBS[match_key] = {
            "status": "running",
            "started_at": time.time(),
            "finished_at": None,
        }

    _RENDER_POOL.submit(_run_render, match_key, venv_python, script_dir)
    return {"match_key": match_key, "status": "running"}


@app.get("/scouting/render/{match_key}")
def scouting_render_status(match_key: str) -> dict:
    """Status of a queued/running/finished render. 'idle' if never requested."""
    with _RENDER_LOCK:
        job = _RENDER_JOBS.get(match_key)
        job = dict(job) if job else None
    if job is None:
        exists = (SCOUTING_ROOT / match_key / "full_match_render.mp4").exists()
        return {"match_key": match_key, "status": "done" if exists else "idle"}
    if job["status"] == "running":
        job["elapsed_s"] = round(time.time() - job["started_at"], 1)
    return {"match_key": match_key, **job}


# ── Vision (vision output) ────────────────────────────────────────────────


@app.get("/vision/team/{team}")
def vision_team(team: int, event_prefix: str = "2026hop") -> dict:
    """Aggregated vision stats for one team across an event. 6h cache."""
    result = get_team_vision(team, event_prefix)
    if result is None:
        raise HTTPException(404, "No vision output directory found")
    return result


@app.get("/vision/match/{match_key}")
def vision_match(match_key: str) -> dict:
    """Full vision output for one match (all 6 robots)."""
    result = get_match_vision(match_key)
    if result is None:
        raise HTTPException(404, f"No vision output for {match_key}")
    return result


class NotesRequest(BaseModel):
    red_teams: list[int]
    blue_teams: list[int]
    event_prefix: str = "2026hop"


@app.post("/vision/notes")
def vision_notes(req: NotesRequest) -> dict:
    """Auto-generate scouting notes from vision data. Empty string if no data."""
    text = build_scouting_notes(req.red_teams, req.blue_teams, req.event_prefix)
    return {"notes": text}


# Cache the 30s per-robot clips on disk so ffmpeg only runs once per (team, match).
_CLIPS_ROOT = Path(__file__).resolve().parent.parent / "clips_cache"
_CLIPS_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/vision/snapshot/{team}")
def vision_snapshot(team: int, event_prefix: str = "2026hop") -> dict:
    """Quick pre-match snapshot for one team.

    Returns aggregated vision stats + most-recent match key + a URL for a 30s
    highlight clip cut from that match's render.
    """
    v = get_team_vision(team, event_prefix)
    if v is None or v.get("matches_count", 0) == 0:
        raise HTTPException(404, f"No vision data for team {team}")

    # Most recent match numerically (qm128 > qm12; sort by trailing number).
    def _sort_key(match_key: str) -> tuple[int, int]:
        # e.g. "2026hop_qm28" -> (0, 28); "2026hop_sf1m1" -> (1, 1)
        tail = match_key.rsplit("_", 1)[-1]
        import re

        m = re.search(r"(\d+)", tail)
        n = int(m.group(1)) if m else 0
        stage = 0 if tail.startswith("qm") else 1
        return (stage, -n)

    matches_sorted = sorted(v.get("matches_seen", []), key=_sort_key)
    recent = matches_sorted[0] if matches_sorted else None

    return {
        "team": team,
        "matches_count": v["matches_count"],
        "avg_balls_scored": v["avg_balls_scored"],
        "max_balls_scored": v["max_balls_scored"],
        "id_confidence": v["id_confidence"],
        "recent_match": recent,
        "clip_url": f"/scouting/clip/{team}/{recent}" if recent else None,
    }


@app.get("/scouting/clip/{team}/{match_key}")
def scouting_clip(team: int, match_key: str):
    """30-second highlight of one robot cut from the full match render.

    Uses ffmpeg to slice the first 30s where the robot is most active. For now
    just takes seconds 20-50 (auto → early teleop). Cached on disk.
    """
    import subprocess

    clip_path = _CLIPS_ROOT / f"{match_key}_{team}.mp4"
    if clip_path.exists() and clip_path.stat().st_size > 0:
        return FileResponse(clip_path, media_type="video/mp4")

    source = SCOUTING_ROOT / match_key / "full_match_render.mp4"
    if not source.exists():
        raise HTTPException(404, f"No render for {match_key} — trigger it first")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(500, "ffmpeg not installed on host")

    # 30-second slice starting at 15s (auto + early teleop). Re-encode to 720p
    # at ~800kbps so 6 clips per match don't push hundreds of MB to the browser.
    # -ss BEFORE -i uses fast seek; slight A/V drift is fine for a preview.
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-ss",
                "15",
                "-i",
                str(source),
                "-t",
                "30",
                "-vf",
                "scale=720:-2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-an",
                "-movflags",
                "+faststart",
                str(clip_path),
            ],
            capture_output=True,
            timeout=90,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"ffmpeg failed: {e.stderr.decode()[-300:]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "ffmpeg timed out clipping video")

    return FileResponse(clip_path, media_type="video/mp4")


@app.get("/team/{team}")
def team(team: int, year: int = 2026) -> dict:
    stats = data_source.get_team_stats(team, year)
    if stats is None:
        if data_source.provider_name() == "tba" and not data_source.is_configured():
            raise HTTPException(400, "No TBA API key configured (set TBA_AUTH_KEY).")
        if not data_source.is_reachable():
            raise HTTPException(503, f"{data_source.provider_name()} data provider unreachable.")
        raise HTTPException(404, f"No {year} data for team {team}.")
    return {
        "stats": stats,
        "archetype": classify_archetype(stats),
        "volatility": profile_volatility(stats),
    }


@app.post("/predict")
def predict(req: MatchupRequest) -> dict:
    """Quant-only: prediction + per-alliance archetype/volatility/chemistry. Fast."""
    prediction = predict_match(req.red_teams, req.blue_teams, year=req.year)
    brief = build_strategy_brief(prediction, notes=req.notes)
    return {
        "prediction": prediction,
        "red": brief["red"],
        "blue": brief["blue"],
        "my_team": get_my_team(),
    }


# ── Streaming reads (NDJSON) ─────────────────────────────────────────────────────


@app.post("/roles")
def roles(req: MatchupRequest) -> dict:
    """Assign each of the 6 robots a short main-job phrase (fast bot model)."""
    _require_ollama()
    prediction = predict_match(req.red_teams, req.blue_teams, year=req.year)
    brief = build_strategy_brief(prediction, notes=req.notes)
    try:
        assignments = generate_roles(prediction, brief, model=req.model)
    except Exception as e:
        raise HTTPException(502, f"Role assignment failed: {e}")
    return {"roles": assignments}


@app.post("/analyze/stream")
def analyze_stream(req: MatchupRequest) -> StreamingResponse:
    """Stream the neutral strategy-captain read (thinking + content events)."""
    _require_ollama()
    prediction = predict_match(req.red_teams, req.blue_teams, year=req.year)
    brief = build_strategy_brief(prediction, notes=req.notes)
    return _stream_read(SYSTEM_PROMPT, brief["text"], req.model or SYNTHESIS_MODEL)


@app.post("/coach/stream")
def coach_stream(req: CoachRequest) -> StreamingResponse:
    """Stream a coaching read for my_team (from body or saved setting)."""
    _require_ollama()
    my_team = req.my_team if req.my_team is not None else get_my_team()
    if my_team is None:
        raise HTTPException(400, "No team set (pass my_team or save one in settings).")
    prediction = predict_match(req.red_teams, req.blue_teams, year=req.year)
    brief = build_strategy_brief(prediction, notes=req.notes)
    try:
        system, user = build_coach_brief(prediction, brief, my_team)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _stream_read(system, user, req.model or SYNTHESIS_MODEL)


@app.post("/captain/stream")
def captain_stream(req: CaptainRequest) -> StreamingResponse:
    """Stream an alliance captain's briefing (game plan + per-robot roles + weakness)."""
    _require_ollama()
    prediction = predict_match(req.red_teams, req.blue_teams, year=req.year)
    brief = build_strategy_brief(prediction, notes=req.notes)

    side = req.our_side if req.our_side in ("red", "blue") else None
    if side is None:
        mt = get_my_team()
        if mt in prediction["red_teams"]:
            side = "red"
        elif mt in prediction["blue_teams"]:
            side = "blue"
        else:
            side = "red"
    system, user = build_captain_brief(prediction, brief, side)
    return _stream_read(system, user, req.model or SYNTHESIS_MODEL)


@app.post("/bots/stream")
def bots_stream(req: MatchupRequest) -> StreamingResponse:
    """Stream a per-robot read for all 6 teams (fast bot model), tagged by team."""
    _require_ollama()
    prediction = predict_match(req.red_teams, req.blue_teams, year=req.year)
    teams = [(s, "red") for s in prediction["red_stats"]] + [
        (s, "blue") for s in prediction["blue_stats"]
    ]

    def gen():
        try:
            for stats, side in teams:
                yield _event(
                    "bot_start", team=stats["team"], name=stats.get("name", "Unknown"), side=side
                )
                for kind, text in stream_bot_analysis(stats):
                    if kind == "content":
                        yield _event("content", team=stats["team"], text=text)
                yield _event("bot_done", team=stats["team"])
            yield _event("done")
        except Exception as e:
            yield _event("error", text=str(e))

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/analyze")
def analyze(req: MatchupRequest) -> dict:
    """Full non-streaming analysis (prediction + 35b read). Slow."""
    _require_ollama()
    prediction = predict_match(req.red_teams, req.blue_teams, year=req.year)
    try:
        result = synthesize(prediction, notes=req.notes, model=req.model)
    except Exception as e:
        raise HTTPException(502, f"Ollama synthesis failed: {e}")
    return {"prediction": prediction, "model": result["model"], "read": result["read"]}


# ── Static frontend (single-container deploy) ─────────────────────────────────
# When frontend/dist exists (built React SPA), serve it at the root so one
# container serves both API and UI. Skipped in dev (Vite serves the frontend).
_frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    # /api/* stays on the FastAPI routes above; everything else falls to the SPA.
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
