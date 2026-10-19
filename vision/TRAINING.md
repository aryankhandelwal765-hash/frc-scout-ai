# Training a real robot detector

The motion-hotspot estimator can't actually see robots — it tracks ball churn and
guesses alliance by field-half, both wrong. This trains a YOLO model that detects
robots and reads alliance from **bumper color**, which is the real fix.

## The 3 steps

### 1. Extract frames  ✅ (already done — 120 frames in `dataset/images/`)
```bash
python extract_frames.py --input test_clip.mp4 --n 120 --outdir dataset/images
```
Samples the cropped top-field across the match, skipping near-duplicates.

### 2. Annotate (this is the work — ~30–60 min)
```bash
python annotate.py --images dataset/images          # run from a Terminal
```
For each frame, **drag a box around every robot** and tag its bumper color:
- `b` / `r` — set the label to BLUE / RED before/while drawing
- `t` toggle last box's color · `u` undo · `c` clear frame
- `n` / SPACE next · `p` prev · `q` save & quit (resumable — labels save as you go)

Tips: label the visible robot body+bumper; skip robots that are mostly off-frame
or fully buried. A frame with genuinely no robots visible is fine to leave empty
(still press `n` so it's saved as a background example). ~120 frames × ~6 robots
≈ 700+ boxes — enough for a first detector on this fixed camera.

### 3. Train
```bash
python train_detector.py --dataset dataset --epochs 120
```
Fine-tunes YOLOv8-nano on Apple-Silicon GPU (MPS) if available. Output:
`runs/detect/frc_robots/weights/best.pt` — your detector. Add `--device cpu` if
MPS errors; `--imgsz 1280` for higher accuracy (slower).

## Then what
Once `best.pt` exists, the detector replaces the motion proxy:
- real robot boxes (not ball piles),
- alliance from bumper color (not field-half),
- track each robot, measure its scoring activity, apportion the scoreboard total.

That's when "Blue robot A/B/C" becomes trustworthy. Ping me with the trained
weights and I'll wire the detector-based estimator.

## Honest expectations
- A first pass on one match/one camera gives a decent detector for *that* feed.
- It won't read team numbers itself — the detector only outputs alliance. Numbers
  ARE recoverable though: at 1080p `match_pipeline/read_numbers.py` reads them with
  EasyOCR and snaps to the six teams TBA says are playing. (At 720p they were
  genuinely unreadable, which is why older notes say identity is impossible.)
- Bulk-dump attribution is still hard, but now built on real robots, not proxies.

## Pre-labelling (faster than drawing every box)

Instead of annotating from scratch, draft labels over many matches and correct them:

```bash
python prelabel.py --matches 40 --per-video 15        # -> dataset_v2/ (images, labels, previews)
python annotate.py --images dataset_v2/images --order dataset_v2/review_order.txt
```

`prelabel.py` runs the fine-tuned detector at two scales, drops boxes on refs/FTA/drivers
(COCO person veto), merges nested duplicates, and fills unreadable bumper colours by the
3-robots-per-alliance rule. Review order puts the frames most likely to need fixes first.
YOLO-World was tried as a zero-shot prelabeler and found useless here (no "robot" boxes).
`review/` holds an optional AI reviewer that proposes keep/delete/recolour per box.
