"""
Phase 5a: LLM interpretation of each robot's auto zone sequence.

For each match × robot:
  1. Extract zone sequence from smoothed_auto_paths.json (falls back to auto_paths.json)
  2. Compute start_delay_s from the trajectory
  3. Call Ollama with few-shot examples from training/auto_sequences.jsonl
  4. LLM returns JSON: {name, description, start_side, delayed, key_zones}
  5. Write results/<match>/auto_interp.json

Run:
  python interpret_auto.py                      # all matches
  python interpret_auto.py --only 2026hop_qm27  # one match
  python interpret_auto.py --force              # redo even if auto_interp.json exists
  python interpret_auto.py --model qwen3:8b     # use smaller/faster model
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "strategy")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, STRATEGY_DIR)

import topdown as td
import zones
from topdown_paths import _fresh, to_canonical_fi

RESULTS = os.path.join(HERE, "results")
TRAINING = os.path.join(HERE, "training", "auto_sequences.jsonl")

DEFAULT_MODEL = "qwen3.6:35b-a3b"
FALLBACK_MODEL = "qwen3:8b"
TIMEOUT = 90


def _load_generate():
    try:
        from scout.engine.ollama_client import generate
        return generate
    except ImportError:
        raise SystemExit(
            f"Cannot import ollama_client from {STRATEGY_DIR}\n"
            "Ensure strategy/ directory exists and 'ollama serve' is running.")


def _load_few_shot(n=5):
    """Load up to n diverse labelled examples as few-shot prompt material."""
    if not os.path.exists(TRAINING):
        return []
    examples, seen_sides = [], set()
    for line in open(TRAINING):
        try:
            r = json.loads(line)
        except Exception:
            continue
        lb = r.get("label")
        if not lb or not r.get("zone_seq"):
            continue
        side = lb.get("start_side", "centre")
        if side not in seen_sides or len(examples) < 3:
            examples.append(r)
            seen_sides.add(side)
        if len(examples) >= n:
            break
    return examples


def _build_system(few_shot):
    lines = [
        "You are an FRC scouting analyst interpreting robot autonomous paths.",
        "Zone sequences come from broadcast-video tracking and may have gaps.",
        "Respond ONLY with valid JSON, no markdown, no extra text:",
        '{"name": "...", "description": "...", "start_side": "left|centre|right",'
        ' "delayed": true|false, "key_zones": [...]}',
        "",
        "Rules:",
        "- name: short scouting shorthand (e.g. 'Left 3-Piece', 'Centre Delayed', 'Right 1-Piece')",
        "- description: 1-2 sentences, concrete facts only. No speculation about game pieces.",
        "- start_side: from the robot's OWN driver-station perspective. Use EXACTLY 'left', 'centre', or 'right' (no other spellings)",
        "- delayed: true when start_delay_s > 1.5 seconds",
        "- key_zones: ordered list of significant zones; omit brief transits",
    ]
    if few_shot:
        lines.append("")
        lines.append("Examples from this event's labelled data:")
        for ex in few_shot:
            zs = " → ".join(ex["zone_seq"]) or "(none)"
            lb = ex["label"]
            ex_out = {
                "name": lb["name"],
                "description": lb["description"],
                "start_side": lb["start_side"],
                "delayed": lb["delayed"],
                "key_zones": ex["zone_seq"],
            }
            lines.append(
                f'\nInput: zones="{zs}" | delay={ex["start_delay_s"]}s'
                f' | alliance={ex.get("alliance", "blue")}'
            )
            lines.append(f"Output: {json.dumps(ex_out)}")
    return "\n".join(lines)


def _build_user(match, station, team, alliance, zone_seq, start_delay_s, duration_s):
    team_num = team.replace("frc", "#") if team else "unknown"
    zs = " → ".join(zone_seq) if zone_seq else "(no zones detected)"
    return (
        f"Team: {team} ({team_num})  Match: {match}  "
        f"Alliance: {alliance}  Station: {station}\n"
        f"Zone sequence: {zs}\n"
        f"Start delay: {start_delay_s}s  Auto duration: {duration_s}s"
    )


def _start_delay_s(pts_with_t, threshold_in=30.0):
    if len(pts_with_t) < 2:
        return 0.0
    t0, x0, y0 = pts_with_t[0]
    for t, x, y in pts_with_t[1:]:
        if ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 > threshold_in:
            return round(t - t0, 2)
    return round(pts_with_t[-1][0] - t0, 2)


def _parse_llm_json(raw):
    """Extract JSON from LLM response, tolerating markdown fences and leading text."""
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if part.startswith("json"):
                part = part[4:]
            part = part.strip()
            try:
                return json.loads(part)
            except Exception:
                pass
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except Exception:
            pass
    return None


def _interpret_robot(match, station, o, mapper, generate, system_prompt, model):
    """Interpret one robot's path. Returns dict with LLM fields + raw data."""
    team = o.get("team")
    path = o.get("path", [])

    if o.get("coords") == "field_in":
        fi = np.array([[fx, fy] for _t, fx, fy in path], float)
        pts = [(row[0], row[1], row[2]) for row in path]
    else:
        canvas = mapper([(c[1], c[2]) for c in path])
        fi = zones.to_field_inches(np.asarray(canvas))
        pts = [(path[i][0], fi[i, 0], fi[i, 1]) for i in range(len(fi))]

    fi_can = to_canonical_fi(fi, o.get("alliance", "blue"))
    zone_seq = zones.zone_sequence(fi_can.tolist(), min_dwell=3)
    delay = _start_delay_s(pts)
    duration = round(path[-1][0] - path[0][0], 2) if len(path) >= 2 else 0.0

    base = {
        "team": team,
        "alliance": o.get("alliance"),
        "source": o.get("source"),
        "zone_seq": zone_seq,
        "start_delay_s": delay,
        "duration_s": duration,
    }

    user_msg = _build_user(match, station, team, o.get("alliance", ""),
                           zone_seq, delay, duration)

    for attempt_model in [model, FALLBACK_MODEL]:
        try:
            raw = generate(model=attempt_model, system=system_prompt,
                           user=user_msg, timeout=TIMEOUT, think=False)
            parsed = _parse_llm_json(raw)
            if parsed and "name" in parsed:
                # Normalize start_side spelling
                if parsed.get("start_side") == "center":
                    parsed["start_side"] = "centre"
                return {**base, **parsed, "model": attempt_model}
        except Exception:
            pass

    # LLM failed: fall back to heuristic
    start_side = "left" if (fi_can[0, 0] < 200) else "right" if (fi_can[0, 0] > 450) else "centre"
    return {
        **base,
        "name": f"{'Delayed ' if delay > 1.5 else ''}{start_side.capitalize()} Auto",
        "description": ("Zone sequence: " + " → ".join(zone_seq)) if zone_seq
                       else "No zones detected — robot may have stayed still or tracking lost.",
        "start_side": start_side,
        "delayed": delay > 1.5,
        "key_zones": zone_seq,
        "model": "heuristic",
    }


def interpret_match(match, mapper, generate, system_prompt, model, force=False):
    out_path = os.path.join(RESULTS, match, "auto_interp.json")
    if not force and os.path.exists(out_path):
        return "skipped"

    smooth_ap = os.path.join(RESULTS, match, "smoothed_auto_paths.json")
    raw_ap = os.path.join(RESULTS, match, "auto_paths.json")

    if os.path.exists(smooth_ap):
        try:
            data = json.load(open(smooth_ap))
        except Exception:
            data = None
    else:
        data = None

    if data is None:
        if os.path.exists(raw_ap) and _fresh(raw_ap):
            try:
                data = json.load(open(raw_ap))
            except Exception:
                return "no_data"
        else:
            return "no_data"

    result = {}
    for station, o in data.get("teams", {}).items():
        team = o.get("team")
        path = o.get("path", [])
        if not team or len(path) < 3:
            result[station] = {"team": team, "error": "too_short"}
            continue
        try:
            result[station] = _interpret_robot(
                match, station, o, mapper, generate, system_prompt, model)
        except Exception as e:
            result[station] = {"team": team, "error": str(e)}

    json.dump(result, open(out_path, "w"), indent=2)
    n_ok = sum(1 for v in result.values() if "name" in v)
    return f"ok:{n_ok}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="limit to these match keys")
    p.add_argument("--force", action="store_true", help="redo even if auto_interp.json exists")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use")
    a = p.parse_args()

    mapper = td.load_mapper()
    if mapper is None:
        raise SystemExit("No field map — run autofit_fieldmap.py first.")

    generate = _load_generate()
    few_shot = _load_few_shot(n=5)
    system_prompt = _build_system(few_shot)
    print(f"Model: {a.model}  |  Few-shot examples: {len(few_shot)}")
    if not few_shot:
        print("  (no training examples — run label_auto.py first for better results)")

    if a.only:
        matches = a.only
    else:
        paths = glob.glob(os.path.join(RESULTS, "*", "auto_paths.json"))
        matches = sorted(os.path.basename(os.path.dirname(q)) for q in paths)

    ok = fail = skip = 0
    for match in matches:
        status = interpret_match(match, mapper, generate, system_prompt, a.model, a.force)
        if status.startswith("ok"):
            ok += 1
            print(f"  {match}: {status}")
        elif status == "skipped":
            skip += 1
        else:
            fail += 1
            print(f"  {match}: {status}")

    print(f"\nDone: {ok} interpreted, {skip} skipped, {fail} failed")
    if skip:
        print("  Use --force to reinterpret already-done matches")


if __name__ == "__main__":
    main()
