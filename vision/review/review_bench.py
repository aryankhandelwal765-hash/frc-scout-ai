"""
Benchmark harness for the box-review task: try models/settings against YOUR corrected
labels, compare accuracy AND speed in one table.

    python review_bench.py init --pre <prelabel dir> --truth <corrected labels dir> \
        --frames <frames.txt>                 # one-time: snapshot an eval set
    python review_bench.py run                # run every experiment in the config
    python review_bench.py run --only qwen3vl_think qwen36_nothink
    python review_bench.py report             # table -> stdout + review_bench_data/REPORT.md
    python review_bench.py report --ref sonnet=<verdicts dir> --ref opus=<verdicts dir>

Experiments live in review_bench_experiments.json (edit freely). Knobs per experiment:
  model            Ollama model tag
  think            true / false / omitted (omitted = don't send the field; needed for
                   models that have no thinking mode)
  num_predict      cap on generated tokens (thinking counts) -- the "cap the thinking" test
  fallback_no_think  if a capped run yields no answer, retry once with think=false
  scale            shrink the image (0.5 = half size) -- the "fewer visual tokens" test
  parallel         concurrent requests (only helps if the Ollama SERVER was started with
                   OLLAMA_NUM_PARALLEL >= this; otherwise it just queues, which is itself
                   a useful measurement)
  prompt           "base" | "keep_bias" (tell the model to keep a box when unsure)
  pull_if_missing  `ollama pull` the model first if it is not installed

Safe to stop and resume at any time: finished frames are skipped. Controls (files in
review_bench_data/):  touch PAUSE  -> wait after in-flight frames;  rm PAUSE -> go on;
touch STOP -> exit cleanly.

Scoring is ai_review.evaluate(), the same one used for the Claude comparison.
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import ai_review as ar
import cv2
from ai_review_local import PROMPT, SCHEMA

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "review_bench_data")
DEFAULT_CONFIG = os.path.join(HERE, "review_bench_experiments.json")
HOST = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

PROMPT_EXTRA = {
    "base": "",
    "keep_bias": (
        "\nImportant: when you are unsure whether a box is a real robot, KEEP it. "
        "Delete only boxes you are confident are not robots (people, ball piles, "
        "field structure) or clear duplicates."
    ),
}


# ---------------------------------------------------------------- init


def cmd_init(args):
    d = args.data
    frames = ar.frame_names(args.frames)
    for sub in ("pre/images", "pre/labels", "truth", "inputs", "runs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    for n in frames:
        shutil.copy2(os.path.join(args.pre, "images", n + ".jpg"), os.path.join(d, "pre/images"))
        shutil.copy2(os.path.join(args.pre, "labels", n + ".txt"), os.path.join(d, "pre/labels"))
        shutil.copy2(os.path.join(args.truth, n + ".txt"), os.path.join(d, "truth"))
    with open(os.path.join(d, "frames.txt"), "w") as fh:
        fh.write("\n".join(frames) + "\n")
    ar.prepare(
        SimpleNamespace(
            pre=os.path.join(d, "pre"),
            frames=os.path.join(d, "frames.txt"),
            out=os.path.join(d, "inputs"),
        )
    )
    print(f"eval set: {len(frames)} frames in {d}")


# ---------------------------------------------------------------- run


def installed_models():
    with urllib.request.urlopen(HOST + "/api/tags", timeout=30) as r:
        return {m["name"] for m in json.load(r)["models"]}


def png_bytes(path, scale):
    img = cv2.imread(path)
    if scale and scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def ask(exp, png, meta, think_override="unset"):
    think = exp.get("think") if think_override == "unset" else think_override
    lines = [
        f"  id {b['id']}: drawn {b['drawn_colour']}, box {b['box_frac']}" for b in meta["boxes"]
    ]
    prompt = PROMPT.format(boxes="\n".join(lines) or "  (no boxes)")
    prompt += PROMPT_EXTRA[exp.get("prompt", "base")]
    options = {"temperature": 0}
    cap = exp.get("num_predict") if think_override == "unset" else None
    if cap:
        options["num_predict"] = cap
    body = {
        "model": exp["model"],
        "stream": False,
        "format": SCHEMA,
        "options": options,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(png).decode()],
            }
        ],
    }
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    msg = out["message"]
    text = msg.get("content") or msg.get("thinking") or ""
    i = text.find("{")
    if i < 0:
        raise ValueError("model returned no JSON object")
    verdict = json.loads(text[i:])
    if not isinstance(verdict, dict) or "boxes" not in verdict:
        raise ValueError("model JSON has no boxes list")
    return verdict, {
        "eval_tokens": out.get("eval_count", 0),
        "think_chars": len(msg.get("thinking") or ""),
    }


def controls(data):
    """Returns False to stop. Blocks while PAUSE exists."""
    pause, stop = os.path.join(data, "PAUSE"), os.path.join(data, "STOP")
    if os.path.exists(stop):
        return False
    if os.path.exists(pause):
        print("paused (rm PAUSE to continue)", flush=True)
        while os.path.exists(pause):
            if os.path.exists(stop):
                return False
            time.sleep(5)
        print("resumed", flush=True)
    return True


def run_experiment(exp, data):
    name = exp["name"]
    rdir = os.path.join(data, "runs", name)
    vdir = os.path.join(rdir, "verdicts")
    os.makedirs(vdir, exist_ok=True)
    json.dump(exp, open(os.path.join(rdir, "config.json"), "w"), indent=1)
    if exp.get("pull_if_missing") and exp["model"] not in installed_models():
        print(f"[{name}] pulling {exp['model']} ...", flush=True)
        subprocess.run(["ollama", "pull", exp["model"]], check=True)
    todo = [
        n
        for n in ar.frame_names(os.path.join(data, "frames.txt"))
        if not os.path.exists(os.path.join(vdir, n + ".json"))
    ]
    print(f"[{name}] {len(todo)} frames to do", flush=True)
    stopped = False

    def work(n):
        nonlocal stopped
        if stopped or not controls(data):
            stopped = True
            return None
        png = png_bytes(os.path.join(data, "inputs", n + ".png"), exp.get("scale", 1.0))
        meta = json.load(open(os.path.join(data, "inputs", n + ".boxes.json")))
        t0 = time.time()
        fallback = False
        try:
            try:
                verdict, st = ask(exp, png, meta)
            except Exception:
                if not exp.get("fallback_no_think"):
                    raise
                fallback = True
                verdict, st = ask(exp, png, meta, think_override=False)
        except Exception as e:
            return {"frame": n, "ok": False, "error": repr(e)[:200], "sec": time.time() - t0}
        json.dump(verdict, open(os.path.join(vdir, n + ".json"), "w"))
        return {"frame": n, "ok": True, "sec": time.time() - t0, "fallback": fallback, **st}

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=exp.get("parallel", 1)) as pool:
        for res in pool.map(work, todo):
            if res is None:
                continue
            with open(os.path.join(rdir, "stats.jsonl"), "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(
                f"[{name}] {res['frame']}  {res['sec']:.0f}s  "
                f"{'ok' if res['ok'] else 'FAIL ' + res.get('error', '')}",
                flush=True,
            )
    # wall time accumulates across resumed sessions (throughput = wall / frames)
    wp = os.path.join(rdir, "wall.json")
    prev = json.load(open(wp))["sec"] if os.path.exists(wp) else 0.0
    json.dump({"sec": prev + time.time() - t_start}, open(wp, "w"))
    return not stopped


def cmd_run(args):
    cfg = json.load(open(args.config))
    exps = [e for e in cfg if not args.only or e["name"] in args.only]
    if not exps:
        raise SystemExit("no experiments matched")
    for exp in exps:
        if not run_experiment(exp, args.data):
            print("STOP requested; rerun `run` to resume.")
            return
    print("all done. python review_bench.py report")


# ---------------------------------------------------------------- report


def load_stats(rdir):
    p = os.path.join(rdir, "stats.jsonl")
    return [json.loads(ln) for ln in open(p)] if os.path.exists(p) else []


def metrics(data, vdir):
    m = ar.evaluate(
        SimpleNamespace(
            pre=os.path.join(data, "pre"),
            truth=os.path.join(data, "truth"),
            verdicts=vdir,
            frames=os.path.join(data, "frames.txt"),
        )
    )
    pv = m["per_verdict"]

    def pct(a, b):
        return f"{a / b:.0%}" if b else "-"

    return {
        "agree": m["dec_ok"] / m["dec_n"] if m["dec_n"] else 0.0,
        "keep": pct(*pv["keep"]),
        "delete": pct(*pv["delete"]),
        "recolor": pct(*pv["recolor"]),
        "prec": m["tp"] / max(1, m["tp"] + m["fp"]),
        "rec": m["tp"] / max(1, m["tp"] + m["fn"]),
        "missing": m["missing_files"],
    }


def cmd_report(args):
    frames = ar.frame_names(os.path.join(args.data, "frames.txt"))
    rows = []
    runs = os.path.join(args.data, "runs")
    for name in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
        rdir = os.path.join(runs, name)
        vdir = os.path.join(rdir, "verdicts")
        if not os.path.isdir(vdir) or not os.listdir(vdir):
            continue
        cfg = json.load(open(os.path.join(rdir, "config.json")))
        st = [s for s in load_stats(rdir) if s["ok"]]
        wall = (
            json.load(open(os.path.join(rdir, "wall.json")))["sec"]
            if os.path.exists(os.path.join(rdir, "wall.json"))
            else 0
        )
        done = len(os.listdir(vdir))
        rows.append(
            {
                "name": name,
                "model": cfg["model"],
                "setup": ",".join(
                    f"{k}={cfg[k]}"
                    for k in ("think", "scale", "num_predict", "parallel", "prompt")
                    if k in cfg
                ),
                "done": done,
                "spf": f"{wall / done:.0f}s" if wall and done else "-",
                "tok": f"{sum(s['eval_tokens'] for s in st) / len(st):.0f}" if st else "-",
                **metrics(args.data, vdir),
            }
        )
    for ref in args.ref or []:
        label, path = ref.split("=", 1)
        rows.append(
            {
                "name": label,
                "model": "(reference)",
                "setup": "",
                "done": len([f for f in os.listdir(path) if f.endswith(".json")]),
                "spf": "-",
                "tok": "-",
                **metrics(args.data, path),
            }
        )
    rows.sort(key=lambda r: -r["agree"])
    head = (
        "| experiment | model | setup | frames | agree | keep | delete | recolor | "
        "precision | recall | sec/frame | tok/frame |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [head] + [
        f"| {r['name']} | {r['model']} | {r['setup']} | {r['done']}/{len(frames)} | "
        f"{r['agree']:.1%} | {r['keep']} | {r['delete']} | {r['recolor']} | "
        f"{r['prec']:.0%} | {r['rec']:.0%} | {r['spf']} | {r['tok']} |"
        for r in rows
    ]
    out = "\n".join(lines)
    print(out)
    open(os.path.join(args.data, "REPORT.md"), "w").write(out + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("--pre", required=True)
    a.add_argument("--truth", required=True)
    a.add_argument("--frames", required=True)
    a.set_defaults(fn=cmd_init)
    b = sub.add_parser("run")
    b.add_argument("--config", default=DEFAULT_CONFIG)
    b.add_argument("--only", nargs="*")
    b.set_defaults(fn=cmd_run)
    c = sub.add_parser("report")
    c.add_argument("--ref", action="append", help="label=verdicts_dir (e.g. a Claude run)")
    c.set_defaults(fn=cmd_report)
    args = p.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
