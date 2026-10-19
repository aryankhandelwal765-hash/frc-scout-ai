"""
Run the ai_review verdict task on a LOCAL vision model through Ollama.

    python ai_review_local.py --model qwen3.6:35b-a3b --frames frames.txt \
        --inputs ai_review/in --out ai_review/verdicts_qwen36

Pause / stop / resume (safe at any time; finished frames are never redone):
    touch <out>/PAUSE     stop after the current frame, wait
    rm <out>/PAUSE        continue
    touch <out>/STOP      exit cleanly; run the same command again to resume
    Ctrl+C                also safe; run the same command again to resume

Reads the numbered <frame>.png + <frame>.boxes.json written by `ai_review.py prepare`
and writes the same verdict JSON the Claude reviewers write, so `ai_review.py score`
grades it identically.
"""

import argparse
import base64
import json
import os
import time
import urllib.request

PROMPT = """You are reviewing auto-generated robot bounding boxes on an FRC (FIRST Robotics) match frame. The image is a 1920x554 overhead-ish view of the field with NUMBERED boxes drawn on it. Box colours are the auto-detector's guess (blue box = guessed blue alliance, orange-red box = guessed red alliance).

Robots are boxy machines with a coloured bumper strip (red or blue) low on the body, usually with a 4-digit team number. Yellow balls are not robots. Big blue or red towers/ramps are field structures, not robots. Referees (striped shirts), field staff (dark shirts) and drivers are people, not robots. Each match has 3 red + 3 blue robots but some may be hidden or cut off.

Box details (ids, drawn colour, position as 0..1 fractions of width/height):
{boxes}

For EVERY numbered box decide:
- keep: box is on a real robot and its colour matches the robot's bumper.
- recolor: real robot but drawn colour is wrong; give the right colour in "colour".
- delete: not a robot (person, ball pile, tower/ramp/structure, floor), OR a duplicate (two boxes on the same robot: delete the worse fit), OR only a tiny sliver of a robot at the frame edge.
Partially occluded robots you can clearly recognise stay. Two overlapping boxes are fine when they cover two DIFFERENT robots. If a bumper colour is not visible, infer it: an alliance has exactly 3 robots, so if 3 robots are clearly blue, any other robot is red (and vice versa). Set unsure true if the frame is crowded or you hesitated on any box.
Also list clearly visible robots with NO box under "missed" (colour blue/red/unknown, box as 0..1 fractions [x1,y1,x2,y2]). Never list slivers.
Answer with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["keep", "delete", "recolor"]},
                    "colour": {"type": "string", "enum": ["blue", "red"]},
                },
                "required": ["id", "verdict"],
            },
        },
        "missed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "colour": {"type": "string", "enum": ["blue", "red", "unknown"]},
                    "box": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["colour", "box"],
            },
        },
        "unsure": {"type": "boolean"},
    },
    "required": ["boxes", "missed", "unsure"],
}


def ask(host, model, image_path, boxes_json, think):
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    meta = json.load(open(boxes_json))
    lines = [
        f"  id {b['id']}: drawn {b['drawn_colour']}, box {b['box_frac']}" for b in meta["boxes"]
    ]
    body = {
        "model": model,
        "stream": False,
        "think": think,
        "format": SCHEMA,
        "options": {"temperature": 0},
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(boxes="\n".join(lines) or "  (no boxes)"),
                "images": [b64],
            }
        ],
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    msg = out["message"]
    # some thinking models (qwen3-vl) ignore think=false and put the schema-constrained
    # answer in the "thinking" field, leaving "content" empty
    text = msg.get("content") or msg.get("thinking") or ""
    i = text.find("{")
    if i < 0:
        raise ValueError("model returned no JSON object")
    v = json.loads(text[i:])
    if not isinstance(v, dict) or "boxes" not in v:
        raise ValueError("model JSON has no boxes list")
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--inputs", required=True, help="dir from `ai_review.py prepare`")
    p.add_argument("--out", required=True)
    p.add_argument("--host", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--think", action="store_true", help="let thinking models reason (slow)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = [ln.strip() for ln in open(args.frames) if ln.strip()]
    pause_file = os.path.join(args.out, "PAUSE")
    stop_file = os.path.join(args.out, "STOP")
    t0 = time.time()
    for i, n in enumerate(names, 1):
        dst = os.path.join(args.out, n + ".json")
        if os.path.exists(dst):
            continue
        # controls, checked between frames so no half-finished frame is ever lost:
        #   touch <out>/PAUSE   wait here until the file is removed
        #   touch <out>/STOP    exit cleanly (rerun the same command to resume)
        if os.path.exists(stop_file):
            print("STOP file found; exiting. Rerun the same command to resume.", flush=True)
            return
        if os.path.exists(pause_file):
            print("paused (remove the PAUSE file to continue)", flush=True)
            while os.path.exists(pause_file):
                if os.path.exists(stop_file):
                    print("STOP file found; exiting.", flush=True)
                    return
                time.sleep(5)
            print("resumed", flush=True)
        try:
            v = ask(
                args.host,
                args.model,
                os.path.join(args.inputs, n + ".png"),
                os.path.join(args.inputs, n + ".boxes.json"),
                args.think,
            )
        except Exception as e:  # keep going; a missing file just scores as "keep"
            print(f"[{i}/{len(names)}] {n}: FAILED {e!r}")
            continue
        json.dump(v, open(dst, "w"))
        print(f"[{i}/{len(names)}] {n}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
