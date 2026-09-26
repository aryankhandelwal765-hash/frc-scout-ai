# examples

One event's real pipeline output, committed so the repo is readable without
running the pipeline or downloading ~20 GB of match video.

Everything here was produced by `vision/match_pipeline`. It is **output**, not
config — nothing in the codebase reads from this directory. Regenerating it
overwrites the working copies under `vision/match_pipeline/`, which are
gitignored.

## `2026hop/`

The 2026 Hopper division. 141 matches were processed; 112 produced both an
`estimate.json` and an `identity.json` and are the ones the strategy side reads.

| File | Produced by | What it is |
|------|-------------|------------|
| `truth.json` | `download_matches.py` | TBA ground truth for every match, used for cross-checking |
| `report.json` / `report.csv` | `compare_tba.py` | Vision estimate vs TBA, per alliance per match |
| `team_routes.json` | `team_routes.py` | Per-team auto routes, clustered by start position |
| `teleop_facts.json` | `teleop_paths.py` | Per-team teleop zone facts |
| `auto_start.json` | `auto_start.py` | Detected auto-start frame per match |
| `scout_table.csv` | `scout_table.py` | One row per (match, team) — the table exported to the strategy half |
| `fuel_split.csv` | `fuel_split.py` | Per-robot fuel estimate from the alliance total |
| `steals.csv` | `steals.py` | Cross-half excursions |
| `path_quality.csv` | `validate_paths.py` | Illegal centre-line crossings in auto — a lower bound on error, not a path-accuracy measure. Regenerated against the live field map; 670 paths over 112 of 139 matches |
| `reports/` | `report_llm.py` | Generated facts-only scouting notes |

### `match_results/2026hop_qm14/`

One representative match, the same layout `vision/match_pipeline/results/<match>/`
has. `auto_view.png` and the six `zone_path_*.png` renders are omitted for size.

| File | What it is |
|------|------------|
| `tracks.json` / `trajectory.csv` | ByteTrack output — per-frame robot boxes and tracks |
| `identity.json` | track id → team number, from bumper-number OCR voting |
| `estimate.json` | Per-robot ball-scoring estimate + activity |
| `auto_paths.json` / `auto_interp.json` | Extracted auto paths and their LLM interpretation |
| `board_series.csv` | Scoreboard OCR time series |
| `auto_paths.png` / `topdown_auto.png` | Rendered top-down auto paths |
