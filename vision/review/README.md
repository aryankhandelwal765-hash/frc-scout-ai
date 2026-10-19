# AI box review

Tooling for cutting down the hand-labelling work. `../prelabel.py` drafts boxes, a vision
model says keep / delete / recolour per box, and a human only draws what is missing.
Nothing here writes to your label files: verdicts go to their own folders, and you decide
what to apply.

| file | what it does |
|---|---|
| `ai_review.py` | `prepare` renders numbered-box images; `score` grades verdicts against human labels |
| `ai_review_local.py` | runs the review on a local Ollama model (supports PAUSE/STOP files) |
| `ai_review_mlx.py` | same task on an MLX model (Apple-silicon native, needs `mlx-vlm`) |
| `review_bench.py` | runs many model/setting experiments against one eval set, one comparison table |
| `review_bench_experiments.json` | the experiment list (model, think, token cap, image scale, prompt) |

## Measured so far (17 hand-corrected frames, 130 boxes, agreement with your calls)

Opus 5.5 91.5%, Sonnet 5.5 86.2%, Qwen3.6-35B-A3B (Ollama, thinking off) 64.6%,
Qwen3-VL-30B (thinking, 10 frames) 68.7%. A small sample, so a few points is noise.
No model found any of the robots a human had to add; drawing new boxes stays manual.

## Workflow

```bash
cd vision
venv/bin/python prelabel.py --matches 40 --per-video 15            # draft labels
venv/bin/python annotate.py --images dataset_v2/images --order dataset_v2/review_order.txt

cd review
python ai_review.py prepare --pre ../dataset_v2 --frames frames.txt --out in
python ai_review_local.py --model qwen3.6:35b-a3b --frames frames.txt --inputs in --out verdicts
python ai_review.py score --pre ../dataset_v2 --truth <corrected labels> --verdicts verdicts --frames frames.txt
```

Bench: `python review_bench.py init ...`, `run`, `report --ref sonnet=<verdicts dir>`.
Data for the bench lands in `review_bench_data/` (gitignored: it holds match frames).

## Pause / stop

Long runs are safe to interrupt. Finished frames are skipped on rerun. Drop a `PAUSE`
file in the output folder to wait after the current frame (`rm` it to continue), or a
`STOP` file to exit cleanly.
