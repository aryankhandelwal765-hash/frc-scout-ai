# FRC Scout AI

Measures FRC robots from match video and fuses that with The Blue Alliance into a strategy brief.

[![CI](https://github.com/aryankhandelwal765-hash/frc-scout-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/aryankhandelwal765-hash/frc-scout-ai/actions/workflows/ci.yml)

## What it does

Scouting an FRC event normally means humans with clipboards watching matches and
typing what they saw. This replaces the watching half with computer vision: a
YOLO detector and ByteTrack follow every robot through the broadcast, bumper
numbers are read with OCR to work out which robot is which team, and an
approximate field map projects paths into field coordinates rather than pixels —
good enough to say which zone a robot worked, not to measure distances.

Those measurements are joined with The Blue Alliance's official results and OPR
aggregates, run through archetype, volatility and chemistry engines, and handed
to a local LLM that writes a strategy brief for a specific matchup. The whole
thing runs on a laptop — the model is served by Ollama, so no match data leaves
the machine.

## Architecture

```mermaid
flowchart LR
    V["Match video<br/>(yt-dlp)"] --> P

    subgraph P["vision/match_pipeline"]
        direction TB
        P1["YOLO detector<br/>+ ByteTrack"] --> P2["fragment stitching"]
        P2 --> P3["bumper OCR<br/>identity voting"]
        P3 --> P4["field map<br/>(polynomial autofit)"]
        P4 --> P5["auto paths · zones<br/>fuel split · steals"]
    end

    P --> EX["export_to_strategy.py<br/>scout_table.csv"]
    EX --> VL["vision_loader.py"]
    TBA[("The Blue Alliance")] --> TL["tba_loader.py"]
    SB[("Statbotics<br/>alternate provider")] -.-> TL

    VL --> SY
    TL --> PR["prediction_engine<br/>archetype · volatility · chemistry"]
    PR --> SY["synthesis_engine<br/>build_strategy_brief()"]
    OL[("Ollama<br/>qwen3:8b")] --> SY

    SY --> API["FastAPI<br/>NDJSON streaming"]
    API --> UI["React dashboard"]

    P -.->|TBA ground truth| CT["compare_tba.py<br/>accuracy check"]
    TBA -.-> CT
```

## Vision pipeline

The measurement half, and the part with the interesting problems. Everything
below lives in `vision/match_pipeline/`.

### Detection and tracking

A YOLOv8 detector fine-tuned on bumper colour finds robots in each frame;
ByteTrack (`bytetrack_frc.yaml`, `trajectories.py`) links them across frames.
Alliance comes from bumper colour, not which half of the field a robot is
standing in — robots cross the field constantly, so field half is not identity.

Static field structures are the main false-positive source: the hub tower's blue
panels and the red ramps read as bumpers. `trajectories.py` builds a mask of the
static field and drops live detections that overlap it.

### Fragment stitching

ByteTrack loses a robot behind traffic and re-acquires it under a new track id.
Measured on `2026hop_qm100`: robot 1706 is read correctly off its bumper on
track 19, track 19 dies, and the later fragments get labelled by rank order — so
the same robot is 1706 for two seconds and 3035 afterwards. The read was right;
it just did not survive the id change.

`stitch.py` links fragment B to fragment A when B starts shortly after A ended,
near where A was *heading*. The gate is velocity-aware: a robot that was moving
keeps moving through the gap, so B's start is compared against A's position
extrapolated forward, and the gate widens with gap length because the
extrapolation degrades. Every distance is a fraction of the field-crop width, so
nothing breaks at a different resolution.

### Identity: OCR voting, not rank order

**Why voting beats rank order.** The obvious way to name robots is driver-station
order: TBA gives each alliance's three team keys in station order, robots line up
by station at auto start, so assign left to right. That is a guess, and it is
wrong whenever the detector misses a robot or two robots swap early. It also
fails silently — you get a confident label with no evidence behind it.

Instead, bumper numbers are read directly (`read_numbers.py`, `identify.py`).
Tesseract reads nothing on bumpers at this scale (measured 0 of 107); EasyOCR
does. Reads are noisy per frame, so they are treated as **votes** keyed by track
id, and the majority across a track wins. Because fragments are stitched first, a
read *anywhere* on a chain names the whole chain.

Frames are sampled per track rather than on a uniform stride. A uniform stride
spends its budget on whichever robot happens to be visible most and starves the
rest.

Identity is **graded**, and every robot records how it was named:

| source | meaning |
|---|---|
| `read` | bumper number actually read |
| `elim` | the only team left once the others were read |
| `rank` | left-to-right station guess, unvalidated |

Station order is the last resort, and it is labelled as such rather than being
laundered into the same confidence as a real read. Across the 670 auto paths
that get validated, **392 (58%) carry a team number read directly off a bumper**;
the remaining 42% are assigned by elimination (112) or rank order (165).

### Calibration

Three approaches were built and measured against each other: a robot-point-cloud
autofit, hand-clicked landmarks, and an AprilTag solve.

**What is in use:** `autofit_fieldmap.py` — a degree-2 polynomial through 9
landmarks inferred from the robot point cloud, written to
`calib.json["fieldmap"]`. Every downstream script reads that map via
`topdown.load_mapper()`. It is good enough for zone-level questions ("did this
robot work the trench?") and not good enough for metric distances.

**What is not:** the AprilTag route. `multi_video_apriltag_calib.py` ran and its
solve was rejected — `calib.json["ground_homography"]` records 16 tags and 64
corners at a **239-pixel reprojection error**, flagged `"bad": true`. Because it
never produced a usable map, `fit_2d_correction.py` never wrote its correction,
and phases 2 through 4 of `overnight_pipeline.py` have never completed. The
scripts are still there because converging that path is the goal, but nothing
downstream consumes their output today. See Roadmap.

### Derived facts

From auto paths in field coordinates the pipeline derives the 22 named field
zones (`zones.py`, `auto_paths.py`), per-team route books (`team_routes.py`), a
per-robot fuel split (`fuel_split.py`), and cross-half excursions
(`steals.py`).

`smooth_trajectories.py` implements Kalman smoothing over the tracks, but it is
part of the un-completed overnight pipeline — no smoothed output exists on disk,
and the derived facts above come from raw ByteTrack paths.

### Validation

Two checks, neither needing hand labels. Both are cross-checks against
independent sources, not accuracy measurements against labelled ground truth —
there is no labelled ground truth for this.

- **Against TBA.** `compare_tba.py` compares the vision estimate with the
  official `score_breakdown` per alliance per match. Numbers under Results.
- **Against the rulebook.** Fully crossing the centre line during auto is a major
  foul, so a path that does is almost certainly a vision error rather than a
  robot error. `validate_paths.py` counts those crossings.

  This is a **lower bound on error, not a measure of path accuracy.** It only
  catches errors large enough to break a rule — a path can be wrong by a metre
  and never cross the line, and that error is invisible to this check. Read it
  as "at least this fraction is wrong", never as "the rest is right".

### "Unmeasured is not zero"

The export to the strategy side (`export_to_strategy.py`) only emits columns that
are actually measured. Climb and defence are not measured by this pipeline, and a
missing value is written as missing rather than 0 — a zero would read downstream
as "this robot never climbs" instead of "we did not look". The generated scouting
notes are facts only, with no human judgement mixed in.

## Results

2026 Hopper division. 140 matches were processed; 112 produced both an
`estimate.json` and an `identity.json`. Numbers below are computed from
`examples/2026hop/report.json`, which is committed — rerun `compare_tba.py` to
reproduce them.

Vision's per-alliance ball count vs TBA's `hubScore.totalCount`:

| | alliance-pairs | matches | MAE | bias (vision − TBA) | Pearson r |
|---|---|---|---|---|---|
| All rows | 278 | 139 | 104.8 | −70.9 | 0.377 |
| Scoreboard OCR produced a reading | 245 | 123 | 37.4 | **+1.0** | **0.862** |

The gap between those rows is the headline finding. In 33 of 278 alliance-pairs
(12%) the scoreboard OCR returned nothing and the estimate is 0, which drags the
aggregate down and produces the large negative bias. Where the OCR reads at all,
correlation is 0.86 and the estimator is essentially unbiased. The failures
cluster in playoff matches (`f1m*`, `sf*`), which use a different broadcast
overlay — a known, bounded problem rather than a diffuse accuracy one.

### Detection recall

Counting distinct track ids per frame and averaging over every frame of all 139
matches:

| | mean | median | min | max |
|---|---:|---:|---:|---:|
| robots tracked per alliance (of 3) | **2.15** | 2.13 | 0.54 | 3.60 |
| robots tracked per match (of 6) | **4.29** | 4.28 | 1.98 | 5.72 |

This is the ceiling on everything downstream. A robot that is never detected
cannot be identified, pathed, or credited with a score.

### Rule violations in auto

`validate_paths.py` counts illegal centre-line crossings. As above: a **lower
bound on error**, not a measure of path accuracy. Computed from the committed
[`examples/2026hop/path_quality.csv`](examples/2026hop/path_quality.csv):

| identity source | violations | n | rate |
|---|---:|---:|---:|
| `read` | 6 | 392 | 1.5% |
| `elim` | 4 | 112 | 3.6% |
| `rank` | 10 | 165 | 6.1% |
| **all** | **20** | **670** | **3.0%** |

The ordering is what matters: paths whose robot was identified by an actual
bumper read break the rule roughly four times less often than paths assigned by
rank order. That is the case for OCR voting over station order, measured rather
than asserted.

Coverage: 670 paths across **112 of 139 matches**. The remaining 27 are 720p-era
geometry, excluded by `topdown_paths._fresh()` because their coordinates predate
the current 1080p crop.

## Strategy

The decision half, in `strategy/scout/engine/`:

- **`archetype_engine`** — playstyle from the EPA breakdown: phase shares and a
  label (auto specialist, teleop workhorse, balanced contributor, …).
- **`volatility_engine`** — floor, ceiling and consistency from `epa_sd` and
  `epa_max`. A high-ceiling, low-floor robot is a different planning problem
  from a metronome with the same average.
- **`chemistry_engine`** — alliance synergy from the three archetypes: phase
  coverage, scoring overlap, and top-heaviness (one robot carrying an alliance
  is fragile).
- **`prediction_engine`** — EPA win probability plus a trained score model.
- **`synthesis_engine`** — merges the quantitative layers with vision-measured
  scoring and any scouting notes into one prompt, then streams a read from a
  local Ollama model. Variants: neutral strategy, alliance captain briefing,
  coaching for your own team, and a short per-robot read for all six.

Reads stream to the browser as NDJSON, including the model's thinking phase, so
the UI shows live progress instead of a frozen spinner.

## Quickstart

Needs Python 3.12 or 3.13 (`.python-version` pins 3.13; CI tests both), Node 24,
and [Ollama](https://ollama.com).
A free [TBA read key](https://www.thebluealliance.com/account) is required for
live data.

### Without Docker (recommended)

This is the path that is actually verified — a fresh clone was taken through it
end to end.

```bash
git clone https://github.com/aryankhandelwal765-hash/frc-scout-ai.git
cd frc-scout-ai

cp .env.example .env          # then put your TBA_AUTH_KEY in it

python3 -m venv strategy/.venv
strategy/.venv/bin/python -m pip install -r strategy/requirements.txt

ollama serve &                 # in its own terminal
ollama pull qwen3:8b

strategy/.venv/bin/python -m uvicorn scout.api.main:app --port 8000 --app-dir strategy &

npm --prefix strategy/frontend install
npm --prefix strategy/frontend run dev
```

The app is at http://localhost:5173, the API docs at http://localhost:8000/docs.
`strategy/start.sh` runs all three together.

Tests:

```bash
strategy/.venv/bin/python -m pip install -r requirements-dev.txt
strategy/.venv/bin/python -m pytest
```

### With Docker (untested)

> **Untested.** The compose file and Dockerfile have been syntax-checked only —
> Docker was not available on the machine this was prepared on, so the stack has
> never been brought up from this branch. Treat the commands below as a starting
> point, not a verified recipe. Use the non-Docker path above for a known-good
> setup.


```bash
git clone https://github.com/aryankhandelwal765-hash/frc-scout-ai.git
cd frc-scout-ai/strategy

cp ../.env.example .env       # then put your TBA_AUTH_KEY in it

npm --prefix frontend install
npm --prefix frontend run build

docker compose up -d --build
docker compose exec ollama ollama pull qwen3:8b
```

The app is at http://localhost/. Set `OLLAMA_MODEL` in `.env` to use a different
model. See [docs/deploy.md](docs/deploy.md) for the full checklist.

### Running the vision pipeline

The pipeline needs its own environment plus ffmpeg and tesseract, and match video
it downloads itself. See [vision/README.md](vision/README.md) and
[vision/match_pipeline/README.md](vision/match_pipeline/README.md).

```bash
python3 -m venv vision/venv
vision/venv/bin/python -m pip install -r vision/requirements.txt
brew install ffmpeg tesseract
```

## Repo layout

```
strategy/           FastAPI + React + LLM synthesis
  scout/engine/     archetype · volatility · chemistry · prediction · synthesis
  scout/api/        NDJSON streaming endpoints
  frontend/         React 19 dashboard
vision/
  classic/          v1 classic-CV tracker (no ML; runs on any clip)
  match_pipeline/   the production pipeline: detect, track, identify, calibrate
  *.py              YOLO detector training toolchain
examples/2026hop/   one event's real pipeline output, committed
tests/              pytest suite, no network
docs/               deployment guide and development notes
```

## Known limitations

- **Detection recall is the ceiling on everything else.** 2.15 of 3 robots per
  alliance are tracked in an average frame. Nothing downstream can recover a
  robot that was never detected.
- **There is no measure of path accuracy.** The centre-line check is a lower
  bound on error — it only catches mistakes big enough to break a rule. A path
  wrong by a metre passes it. Measuring real path accuracy would need labelled
  ground truth, which does not exist for this footage.
- **The field map is approximate, and the AprilTag route has not converged.**
  The map in use is a 9-landmark polynomial autofit — fine for "which zone was
  this robot in", not for distances. The AprilTag multi-video solve was
  attempted and rejected at a 239-pixel reprojection error, so phases 2 through
  4 of `overnight_pipeline.py` (2D correction, Kalman smoothing, re-render)
  have never completed. Nothing downstream uses AprilTag output today.
- **Auto-path validation covers 112 of 139 matches.** The other 27 are 720p-era
  geometry and are skipped by the freshness gate, so the 3.0% figure does not
  cover the whole event.
- **Per-robot splits are estimates; alliance totals are checkable.** The
  scoreboard gives an exact per-alliance count. Dividing it among three robots
  is a heuristic weighted by each robot's scoring-zone activity, and it is only
  as good as the robots are separable in space. When all three are in one
  churning pile, the split is unreliable.
- **Scoring is a heuristic, not a rules engine.** The classic tracker credits a
  ball to whichever robot possessed it last before it entered a goal zone. It
  does not implement the game manual.
- **Scoreboard OCR fails on playoff broadcasts.** 12% of alliance-pairs, almost
  all playoff matches, produce no reading at all — see Results.
- **Win probability is miscalibrated.** `predict_match` calls
  `win_probability_from_diff` with `sigma=100` while OPR is on a roughly
  300-per-team scale, so probabilities peg high on lopsided matchups. This is
  deliberately left alone rather than tuned by eye; a test pins the current value
  so it cannot drift silently. Fixing it properly means normalising OPR onto an
  EPA scale in `feature_engineering.py`.
- **42% of identities are not read off a bumper.** 392 of 670 auto paths carry a
  directly-read team number; 112 come from elimination and 165 from rank order.
  `ALLIANCE_ORDER` has never been validated, so every `rank`-sourced label is a
  guess — and those paths break the centre-line rule four times more often than
  read ones.
- **Endgame is hard-zeroed for 2026.** Climb is worth ~nothing this season, so
  `tba_loader._phase_points` returns 0 for it and the prompts refuse to discuss
  it. That is a season-specific rule, not a general one.

## Roadmap

- Get the AprilTag calibration converging — it currently solves at a 239-pixel
  reprojection error and is rejected. That unblocks the 2D correction and the
  Kalman smoothing phases, and is the prerequisite for any metric claim.
- Normalise OPR onto the EPA scale and retune the win-probability sigma against
  held-out matches.
- Handle the playoff broadcast overlay so scoreboard OCR stops failing on `sf*`
  and `f*` matches.
- Validate `ALLIANCE_ORDER` so `rank` identities carry a real error bar.
- Move the render queue off the in-process worker pool onto something that
  survives an API restart.

## License

MIT — see [LICENSE](LICENSE).
