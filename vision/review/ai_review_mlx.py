"""
Run the ai_review verdict task on a local MLX vision model (Apple-silicon native).

    python ai_review_mlx.py --frames frames.txt --inputs ai_review/in --out verdicts_mlx
    python ai_review_mlx.py ... --limit 2          # smoke test: first 2 frames only

Needs `pip install mlx-vlm` (keep it in its own venv; it pulls a different transformers
than the rest of the project). Writes the same verdict JSON as ai_review_local.py, so
`ai_review.py score` grades it identically, plus <out>/stats.jsonl with per-frame timing.

MLX does not enforce a JSON schema the way Ollama does, so output is validated and a bad
reply is retried once; frames that still fail are logged and left for the next run.

Controls (files in <out>/): touch PAUSE, rm PAUSE, touch STOP. Finished frames are
skipped, so rerunning the same command resumes.
"""

import argparse
import json
import os
import time

from ai_review_local import PROMPT

DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"

# MLX cannot force a JSON schema (Ollama can), so the reply shape is shown in the prompt.
SCHEMA_HINT = (
    "\n\nReply with ONLY a JSON object in exactly this shape, one entry per box id:\n"
    '{"boxes":[{"id":0,"verdict":"keep"},{"id":1,"verdict":"recolor","colour":"red"},'
    '{"id":2,"verdict":"delete"}],"missed":[{"colour":"blue","box":[0.1,0.4,0.2,0.7]}],'
    '"unsure":false}'
)


# Bounded "reasoning budget": thinking stays off, but each box gets a short written
# reason BEFORE its verdict, so the model reasons about every box in ~10 tokens each.
SCHEMA_HINT_REASON = (
    "\n\nFor each box first write a reason of at most 10 words (what is in the box, and "
    "its bumper colour or why it is not a robot), THEN the verdict. Reply with ONLY a "
    "JSON object in exactly this shape, one entry per box id:\n"
    '{"boxes":[{"id":0,"why":"red robot, red bumper visible","verdict":"keep"},'
    '{"id":1,"why":"referee standing, no robot","verdict":"delete"}],'
    '"missed":[{"colour":"blue","box":[0.1,0.4,0.2,0.7]}],"unsure":false}'
)


def parse_verdict(text):
    text = text.split("</think>")[-1]  # thinking runs: the answer follows the reasoning
    i = text.find("{")
    if i < 0:
        raise ValueError("no JSON object in reply")
    v = json.JSONDecoder().raw_decode(text[i:])[0]
    if isinstance(v, dict) and "boxes" not in v:
        # tolerate {"box_0": "delete", ...}: models sometimes invent this shape
        flat = [
            {"id": int(k.split("_")[1]), "verdict": str(x).split("_")[0]}
            for k, x in v.items()
            if k.startswith("box_") and k.split("_")[1].isdigit() and isinstance(x, str)
        ]
        if flat:
            v["boxes"] = flat
    if not isinstance(v, dict) or "boxes" not in v:
        raise ValueError("JSON has no boxes list")
    v.setdefault("missed", [])
    v.setdefault("unsure", False)
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--frames", required=True)
    p.add_argument("--inputs", required=True, help="dir from `ai_review.py prepare`")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=900)
    p.add_argument("--limit", type=int, help="only the first N unfinished frames")
    p.add_argument("--think", action="store_true", help="leave the model's thinking on")
    p.add_argument(
        "--reason",
        action="store_true",
        help="thinking off, but a short written reason per box before each verdict",
    )
    args = p.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    os.makedirs(args.out, exist_ok=True)
    pause_file = os.path.join(args.out, "PAUSE")
    stop_file = os.path.join(args.out, "STOP")
    names = [ln.strip() for ln in open(args.frames) if ln.strip()]

    t0 = time.time()
    model, processor = load(args.model)
    config = load_config(args.model)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    def build_prompt(text):
        try:
            return apply_chat_template(
                processor, config, text, num_images=1, enable_thinking=args.think
            )
        except TypeError:
            return apply_chat_template(processor, config, text, num_images=1)

    done = 0
    for n in names:
        dst = os.path.join(args.out, n + ".json")
        if os.path.exists(dst):
            continue
        if args.limit is not None and done >= args.limit:
            break
        if os.path.exists(stop_file):
            print("STOP file found; exiting.", flush=True)
            return
        while os.path.exists(pause_file):
            if os.path.exists(stop_file):
                return
            print("paused (rm PAUSE to continue)", flush=True)
            time.sleep(5)

        meta = json.load(open(os.path.join(args.inputs, n + ".boxes.json")))
        lines = [
            f"  id {b['id']}: drawn {b['drawn_colour']}, box {b['box_frac']}" for b in meta["boxes"]
        ]
        text = PROMPT.format(boxes="\n".join(lines) or "  (no boxes)") + (
            SCHEMA_HINT_REASON if args.reason else SCHEMA_HINT
        )
        png = os.path.join(args.inputs, n + ".png")

        verdict, stats, err = None, {}, ""
        t1 = time.time()
        for attempt in (1, 2):
            suffix = "" if attempt == 1 else "\nReply with ONLY the JSON object, nothing else."
            res = generate(
                model,
                processor,
                build_prompt(text + suffix),
                image=[png],
                max_tokens=args.max_tokens,
                temperature=0.0,
                verbose=False,
            )
            raw = res.text if hasattr(res, "text") else str(res)
            stats = {
                "prompt_tokens": getattr(res, "prompt_tokens", None),
                "gen_tokens": getattr(res, "generation_tokens", None),
                "gen_tps": getattr(res, "generation_tps", None),
                "attempts": attempt,
            }
            try:
                verdict = parse_verdict(raw)
                break
            except Exception as e:
                err = repr(e)
        sec = time.time() - t1
        done += 1
        rec = {"frame": n, "sec": round(sec, 2), "ok": verdict is not None, **stats}
        if verdict is not None:
            json.dump(verdict, open(dst, "w"))
        else:
            rec["error"] = err
        with open(os.path.join(args.out, "stats.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{n}  {sec:.1f}s  {'ok' if verdict else 'FAILED ' + err}", flush=True)


if __name__ == "__main__":
    main()
