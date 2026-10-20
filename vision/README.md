> **Note:** the v1 classic-CV tracker described below was archived to `vision/_archive/classic/` and is no longer
> maintained. The live pipeline is `match_pipeline/` (YOLO + ByteTrack); see the root `README.md` and `CLAUDE.md`.

# FRC Match Vision — Robot + Ball Scoring Tracker

Tracks robots by alliance color (red/blue bumpers) **and** the game-piece balls
they score, in FRC match video — using classic CV (HSV thresholding + contour /
circularity detection + a nearest-centroid tracker). No training, runs on CPU.

For each ball, it figures out which robot was carrying it and whether it ended
up in a goal, and produces a **per-robot ball-scoring tally**.

```
frame ─► color masks ─► robot blobs + round ball blobs
      ─► per-alliance robot trackers + ball tracker (grace period)
      ─► possession (ball ─► nearest robot) + scoring (ball vanishes in a goal)
      ─► annotated.mp4 + CSVs + summary.json
```

## Files

| file | what it is |
|------|-----------|
| `classic/track_match.py` | main entry point (robots **and** balls **and** scoring) |
| `classic/config.py` | color presets, tunable params, goal-zone type |
| `classic/detection.py` | color masks, robot blobs, circular ball blobs |
| `classic/tracking.py` | nearest-centroid tracker with a grace period |
| `classic/scoring.py` | possession + goal-zone scoring engine |
| `classic/overlay.py` | annotated-video drawing (boxes, balls, scoreboard) |
| `classic/make_test_clip.py` | generates a synthetic clip to validate the pipeline |
| `classic/robot_tracker.py` | the original robots-only script (kept for reference) |

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Sanity-check it works (no footage needed)

The synthetic clip has a known answer — Red robot scores 2 balls, Blue scores 1:

```bash
python classic/make_test_clip.py                              # -> test_clip.mp4, test_zones.json
python classic/track_match.py --input test_clip.mp4 --zones test_zones.json --outdir out
```

You should see `RED 2 / BLUE 1`, and `out/annotated.mp4` shows the boxes, ball
circles, possession lines, live scoreboard, and a "SCORE!" flash on each goal.

## 3. Grab a match clip

This needs to run on your machine (the sandbox can't reach YouTube):

```bash
pip install yt-dlp
yt-dlp -f "best[height<=720]" -o clip.mp4 "PASTE_YOUTUBE_URL_HERE"
```

Pick a **static broadcast angle** (fixed overhead/scoring-table camera), not a
panning/handheld shot — the tracker has no camera-motion compensation. Trim to a
short segment so you iterate fast:

```bash
ffmpeg -i clip.mp4 -ss 00:00:30 -t 00:00:20 -c copy clip_trimmed.mp4
```

## 4. Tell it where the goals are

Scoring = "a ball disappeared inside a goal region." You define those regions
once per camera angle by dragging boxes on the first frame:

```bash
python classic/track_match.py --input clip_trimmed.mp4 --pick-zones --zones-out zones.json
```

Drag a box around each goal/hub, press ENTER after each, ESC when done, and
answer the alliance prompt (`red` / `blue` / `shared`). Or hand-write
`zones.json`:

```json
[
  {"name": "red_goal",  "alliance": "red",    "rect": [20, 200, 120, 140]},
  {"name": "blue_hub",  "alliance": "shared", "rect": [800, 180, 150, 160]}
]
```

`rect` is `[x, y, w, h]` in pixels. `alliance: "shared"` credits whichever robot
last possessed the ball (use it for a single central hub).

## 5. Run it

```bash
python classic/track_match.py --input clip_trimmed.mp4 --zones zones.json --outdir out --ball-color yellow
```

Outputs in `out/`:
- `annotated.mp4` — boxes + ball circles + possession lines + goal zones + live scoreboard
- `robot_positions.csv` — `frame, time_sec, alliance, track_id, x, y, w, h, visible`
- `ball_positions.csv` — `frame, time_sec, ball_id, x, y, w, h, visible, possessor`
- `scoring_events.csv` — `frame, time_sec, ball_id, zone, credited_alliance, credited_robot, x, y`
- `summary.json` — per-robot and per-alliance ball totals

`--ball-color` presets: `yellow` (power cells / fuel, the default and most
reliable), `orange`, `green` (teal/algae), `red`, `blue`. **Note:** red/blue
cargo shares its color with the bumpers, so those lean entirely on the
circularity filter — yellow is far more robust when your game allows it.

## 6. Tuning (broadcast footage is messier)

In rough order of impact — flags override the defaults in `classic/config.py`:

1. **`--robot-min-area` / `--ball-min-area`** — raise to kill crowd/banner noise;
   most junk blobs are small. Balls are much smaller than robots.
2. **Goal zones** — tighten them to just the scoring opening. Too-big zones count
   balls that merely pass in front of the goal.
3. **`--ball-min-circularity`** (0–1) — raise it to reject non-round same-colored
   objects; lower it if real balls get missed (motion blur flattens circles).
4. **`--possession-radius`** — px within which a ball counts as "held" by a robot.
   Too small → shots aren't attributed; too big → wrong robot gets credit.
5. **Narrow the HSV ranges** in `classic/config.py` — sample real bumper/ball pixels from a
   frame (`cv2.imwrite` a frame, open in Preview, use the color picker) and
   tighten the ranges around what you actually see.

## How scoring is decided (and its limits)

- **Possession**: each frame, every visible ball is attributed to the nearest
  robot within `possession_radius` (or inside its bumper box). The *last*
  possessor is remembered — so a ball shot into a goal still credits the shooter
  while it's mid-air.
- **Score**: a ball's track is deleted when it's been missing for
  `ball_grace_frames`. If its last position was inside a goal zone, that's a
  score, credited to the last possessor.

This is a heuristic, not a rules engine: no depth, no field model, no awareness
of actual game scoring rules. It converts pixels into a plausible per-robot
tally. Where it's weakest: balls occluded for a long time, two balls entering a
goal at once, and cargo that shares the bumper color.

## Where this goes next

- Swap the color-blob detector for a fine-tuned **YOLO** model — handles
  occlusion, lighting, and non-color game pieces (rings/coral) far better.
- Add a **homography** (pixel → real field feet) so positions and zones are in
  field coordinates instead of pixels — makes it robust to camera framing and a
  legitimate data source for a scouting ML pipeline.
- Cross-check the vision tally against **The Blue Alliance** match scores to
  auto-measure attribution accuracy.
