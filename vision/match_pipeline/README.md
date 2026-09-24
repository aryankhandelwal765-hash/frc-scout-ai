# match_pipeline — FRC match vision: multi-view fusion, identity, auto paths

> **CURRENT STATE (2026-08-03).** Read this first — sections below the `---` predate
> the 1080p multi-view rewrite and describe a flow that no longer exists.

**Game is 2026 REBUILT.** Field 651.2 x 317.7in. AUTO is **20 seconds** (not 15 —
that bug silently truncated every auto path by a quarter). TELEOP is 2:20, split
TRANSITION 10s + SHIFT 1-4 at 25s each + END GAME 30s, which is exactly what TBA
reports as `hubScore.shift1Count..shift4Count` / `transitionCount` / `endgameCount`.

**Source is 1080p, split into 3 time-aligned views.** `preprocess.py` emits
`videos/<m>.mp4` (top, 1920x712) plus `videos/revL|revR/<m>.mp4`. Frame N lines up
across all three — that alignment is what makes fusion possible. 720p was not
enough: bumper numbers are unreadable there.

**Identity is READ, not guessed.** EasyOCR on bumpers (tesseract reads 0/107 on
them; it is only for the scoreboard). Reads snap to the six teams TBA says are
playing, so a partial read still resolves. Two rules matter and were both bugs:
- Sampling is budgeted PER TRACK, not by a flat stride. A stride spends its reads
  on whatever fragment lives longest and leaves most tracks with one sample — and
  one sample can never identify anything, since `MIN_VOTES` is 2.
- Team uniqueness holds only between tracks alive AT THE SAME TIME. A global
  "six teams, six tracks" cap left every later fragment unnamed however clearly its
  bumper was read. Fixing it took qm27 from 11 to 40 identified tracks.

**ByteTrack fragments hard** — ~160-230 track ids for 6 robots per match. `stitch.py`
chains fragments back into per-robot runs (velocity-aware gating), then a read
anywhere on a chain locks the whole chain.

**Identity is graded, and the grades are measured.** Every robot carries how it was
named, and an independent check — the manual makes fully crossing the CENTER LINE
in AUTO a MAJOR FOUL, so a path that does is almost certainly a vision error —
gives an accuracy estimate with no hand labelling:

| source | meaning | illegal-crossing rate |
|---|---|---|
| `read` | bumper number read | 0.8% |
| `elim` | only team left after the others were read | 3.8% |
| `rank` | left-to-right station guess, UNVALIDATED | 7.1% |

`ALLIANCE_ORDER` has never been validated. Treat every `rank` label as unproven.

**Calibration is approximate and that is a deliberate stopping point.** Three
approaches were measured against each other: the robot-cloud autofit (active),
hand-clicked landmarks, and an AprilTag PnP solve. Autofit won — 3.0% illegal
crossings and 96.5% of points landing on the carpet, vs 27.6%/35.0% for PnP. Good
enough for zone-level questions; not good enough for metric distances. Do not
re-litigate without a measured before/after from `validate_paths.py`.

**Zones are real now.** The manual sizes every structure but places them only in
figures. The official AprilTag layout closes that gap because tags are mounted ON
the structures at published coordinates — that is how `zones.py` knows the HUB
occupies 158.6..205.6in and where the trenches sit.

| file | what |
|------|------|
| `pipeline.py` | per-match chain download→preprocess→trajectory→scoring→identify→auto |
| `identify.py` | per-track identity from bumper reads, top + both reverse views |
| `stitch.py` | chain ByteTrack fragments into per-robot runs; lock identity onto a chain |
| `auto_start.py` | exact AUTO start frame per match, read off the broadcast timer |
| `zones.py` | 22 named field zones in inches; point→zone and path→zone_sequence |
| `steals.py` | teleop excursions into the opponent half and the run home |
| `validate_paths.py` | accuracy check using the CENTER LINE rule as ground truth |
| `scout_table.py` | one row per (match, team) in hand-scouting schema, source-tagged |
| `report_llm.py` | readable scouting notes via local Ollama, facts-only prompt |
| `export_to_strategy.py` | hand-off to the strategy strategy system |
| `multi_video_apriltag_calib.py` + `fit_2d_correction.py` | camera→field map (canonical; older variants in `_archive/`) |
| `topdown.py` / `topdown_paths.py` / `team_routes.py` | bird's-eye field + route books |
| `render_full_match.py` / `render_fusion.py` / `render_autopaths.py` | visual checks |

**Traps that have each cost hours here:**
- Hardcoded 1280x720 pixel boxes. Use `trajectories.field_crop()`,
  `scoring_estimator.SCOREBOARD_FRAC`, `preprocess.TIMER_BOX_FRAC`.
- `TIMER_BOX_FRAC` is a fraction of the FULL frame; the top view is only 0.661 of
  its height, so the y fractions need rescaling or the timer reads nothing.
- TBA serialises "did not climb" as the STRING `"None"`. `bool("None")` is True, so
  a truthiness test scores 840/840 robots as climbers against a real 13/840.
- Filtering stale data by file mtime breaks at midnight. Detect stale geometry from
  the COORDINATE RANGE instead (`topdown_paths._fresh`).
- `auto_paths.json` is keyed by STATION, not team. Read the team off the record.

**Known-open:** teleop paths for `tele_bump`/`tele_trench`; the fuel split
(TBA alliance total x zone-gated activity share); detection recall sits at ~2.3 of
3 robots per alliance, which is the ceiling on everything else.

---

## (historical) TBA cross-check for the vision scoring estimator

Downloads every match video for a TBA event (default **2026hop**, Hopper
division), runs the vision estimator on each, and scores the vision output
against The Blue Alliance's official numbers.

```
TBA API ─► matches (+ score_breakdown + youtube keys)
        ─► truth/<match>.json                     (ground truth)
        ─► yt-dlp ─► videos/<match>.mp4            (download_matches.py)
videos  ─► scoring_estimator (parallel) ─► results/<match>/estimate.json  (batch_run.py)
join    ─► report.csv + report.json               (compare_tba.py)
```

| file | what |
|------|------|
| `tba.py` | TBA API v3 client (key loading, match fetch) |
| `download_matches.py` | fetch matches, save TBA truth, download videos (parallel) |
| `batch_run.py` | run `scoring_estimator` across all videos (process pool) |
| `compare_tba.py` | vision vs TBA per alliance → MAE / bias / correlation |

## Run it from PyCharm (no terminal)

Each script has an **`EDIT ME` CONFIG block** at the top. Change those constants,
then hit the green ▶ Run button. No command-line args needed. Defaults = the
small 5-match test.

**Interpreter:** point PyCharm's project interpreter at `vision/venv`
(Settings → Project → Python Interpreter) — it has opencv, numpy, pytesseract.

### 0. TBA key (once)
Free read-only key from <https://www.thebluealliance.com/account> → "Read API Keys".
Create a file `match_pipeline/.tba_key` containing just the key on one line.
(Or add `TBA_AUTH_KEY=...` under Environment variables in each Run config.)

### 1. Small test first (5 matches)
1. Open `download_matches.py`, ▶ Run. Defaults: `EVENT="2026hop"`, `LIMIT=5`.
   → 5 videos land in `videos/`, all match truth in `truth/`.
2. Open `batch_run.py`, ▶ Run. → `results/<match>/estimate.json` for each.
3. Open `compare_tba.py`. It defaults to `SHOW_BREAKDOWN="2026hop_qm1"` — ▶ Run
   to print that match's real TBA `score_breakdown` and pick the per-alliance
   game-piece field name.
4. In `compare_tba.py`, set `SHOW_BREAKDOWN=""` and `TBA_FIELD="<that field>"`,
   ▶ Run again. → prints MAE/bias/correlation, writes `report.csv`+`report.json`.
   (Leave `TBA_FIELD=None` to compare against total match score — weak proxy.)

### 2. Full run
- `download_matches.py`: set `LIMIT=None` (all) or `QUALS_ONLY=True`, ▶ Run.
- `batch_run.py`: set `WORKERS=12` (48GB / 18-core: go wide), ▶ Run.
- `compare_tba.py`: keep your `TBA_FIELD`, ▶ Run.

Re-running skips already-downloaded videos and already-estimated matches
(set `FORCE=True` in `batch_run.py` to redo).

### Terminal equivalent (optional)
Every constant is also a CLI flag that overrides it, e.g.
`python download_matches.py --event 2026hop --limit 5`,
`python batch_run.py --workers 12`,
`python compare_tba.py --tba-field <field>`.

## Pipeline B — auto paths + robot→team classification (GPU)

Separate from the OCR cross-check. Uses the trained YOLO detector + ByteTrack to
log robot trajectories, slices the auto period, and classifies each robot to a
real TBA team number, then aggregates each team's preferred auto route.

| file | what |
|------|------|
| `trajectories.py` | YOLO+ByteTrack per-frame robot ground positions → `trajectory.csv` (runs on **MPS/GPU**) |
| `batch_trajectories.py` | run the above across all videos (parallel GPU workers) |
| `field_calibrate.py` | **one-time** click the 6 station start-anchors (fixed camera → reused for all 140); optional homography |
| `auto_paths.py` | slice auto, match each robot's start to nearest station anchor → **TBA team**, draw team-labeled paths |
| `batch_auto.py` | run `auto_paths` over all matches, then `preferred_paths` |
| `preferred_paths.py` | average each team's auto route across its matches → per-team + combined overlays |

### How classification works
TBA gives each alliance's 3 team keys **in driver-station order** (blue1/2/3,
red1/2/3). Robots line up by station at auto start. So: calibrate the 6 station
anchors once → for each match, greedily match detected robot start positions to
same-alliance anchors → station → TBA team.

This is the FALLBACK. Since moving to 1080p the bumper numbers are actually
readable, so `identify.py` READS them (see Multi-view identity below) and rank
order is only used for a robot whose number never resolved.

### Run order (PyCharm ▶ each)
1. `download_matches.py` — videos + truth (shared with Pipeline A).
2. `batch_trajectories.py` — `trajectory.csv` per match. GPU; `WORKERS=3` default.
3. *(optional but better)* `batch_run.py` — its `board_series.csv` gives a real
   auto-start frame; without it, auto start falls back to track-presence.
4. `field_calibrate.py` — **once.** Grab an auto-start frame (`--at <sec>`), click
   `blue1,blue2,blue3,red1,red2,red3` where each robot lines up. Saves `calib.json`.
5. `batch_auto.py` — writes `results/<match>/auto_paths.png` + `.json`, and
   `preferred_paths/<team>.png` + `_all_means.png`.

### Verified on 2026hop_qm1
Detector finds `blue_robot`/`red_robot` on real footage; ByteTrack holds IDs;
all 6 robots classified to the correct roster (blue 3175/1710/3512, red
8749/2486/4201). Trajectory logger runs on **device=mps**.

### Caveats
- **Calibration is required and real.** The `calib.json` from the smoke test has
  placeholder anchors — re-run `field_calibrate.py` and click actual start spots,
  or classification/paths are only roughly right.
- **Auto-start frame.** Most reliable when `board_series.csv` exists (run
  `batch_run.py` first). Else pass `--start-frame` per match.
- **Fisheye.** The overhead shot is barrel-distorted, so pixel paths are curved.
  Add a homography in `field_calibrate.py --homography` for metric field paths.
- **Identity is read first, inferred second.** `identify.py` reads bumper numbers
  in the top + both reverse views and votes per track; only tracks with no
  resolved read fall back to start-position↔station order (which is wrong if a
  robot starts far off its station). Measured on 2026hop_qm27: 5 of 6 teams by
  read, 1 by fallback.

## Verified against real 2026hop footage

- **2026 game = FIRST AGE**, a yellow-ball game — the exact "FIRST AGE" 1280×720
  broadcast layout `scoring_estimator.py`'s OCR boxes were tuned for. Confirmed on
  `2026hop_qm1`: OCR reads blue **243** / red **162** correctly from the scoreboard.
- **TBA ball field = `hubScore.totalCount`** (balls scored per alliance; also
  `hubScore.autoCount` / `teleopCount` / `endgameCount` for phase splits). This is
  already the default `TBA_FIELD` in `compare_tba.py`.
- Videos are 1280×720 av1 @ 60fps — OpenCV decodes them fine.
- Per-alliance totals are the checkable quantity. The per-robot A/B/C split is an
  estimate TBA can't validate (TBA has no per-robot piece counts).
- Minor expected gaps: the scoreboard counter recirculates past 360 and TBA logs
  ~2 `uncounted` balls, so vision won't match `totalCount` to the exact ball. MAE
  is the point — that's what `compare_tba.py` measures.
