"""
Phase 5c: Written auto scouting report per team.

Reads:
  team_routes.json          -- cluster metadata (which matches, zone sequences, delay info)
  results/*/auto_interp.json -- per-match LLM descriptions (from interpret_auto.py)

Per team, calls LLM to synthesise a written scouting entry covering each distinct routine.
Outputs:
  auto_book/<team>.md       -- per-team markdown
  auto_book/_all.md         -- all teams combined (sorted by team number)
  auto_book/_all.pdf        -- PDF via pandoc if installed (graceful skip otherwise)

Run:
  python auto_book.py               # all teams
  python auto_book.py --team 8044   # one team
  python auto_book.py --force       # regenerate even if files exist
  python auto_book.py --model qwen3:8b  # faster model
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "strategy")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, STRATEGY_DIR)

RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(HERE, "auto_book")
ROUTES_JSON = os.path.join(HERE, "team_routes.json")

DEFAULT_MODEL = "qwen3.6:35b-a3b"
FALLBACK_MODEL = "qwen3:8b"
TIMEOUT = 120

# Team name lookup from TBA data (best-effort from identity.json)
_TEAM_NAMES = {}


def _load_generate():
    try:
        from scout.engine.ollama_client import generate

        return generate
    except ImportError:
        raise SystemExit(
            f"Cannot import ollama_client from {STRATEGY_DIR}\nEnsure 'ollama serve' is running."
        )


def _load_team_name(team):
    """Try to find a human-readable team name from any available source."""
    if team in _TEAM_NAMES:
        return _TEAM_NAMES[team]
    # Try truth/ directory for team names (stored by some scripts)
    truth_dir = os.path.join(HERE, "truth")
    for f in glob.glob(os.path.join(truth_dir, "*.json")):
        try:
            d = json.load(open(f))
            for t in d.get("blue_teams", []) + d.get("red_teams", []):
                k = t.get("key") or t.get("team_key")
                nm = t.get("nickname") or t.get("name")
                if k and nm:
                    _TEAM_NAMES[k] = nm
        except Exception:
            pass
    return _TEAM_NAMES.get(team, "")


def _label(team):
    return team.replace("frc", "#") if str(team).startswith("frc") else str(team)


def _load_per_match_interps():
    """Load all auto_interp.json files. Returns {team: [(match, station, interp_dict)]}."""
    per_team = defaultdict(list)
    for interp_path in glob.glob(os.path.join(RESULTS, "*", "auto_interp.json")):
        match = os.path.basename(os.path.dirname(interp_path))
        try:
            data = json.load(open(interp_path))
        except Exception:
            continue
        for station, info in data.items():
            team = info.get("team")
            if team and "name" in info:
                per_team[team].append((match, station, info))
    return per_team


def _build_synthesis_prompt(team, team_name, clusters, match_interps):
    """Build the user prompt for the per-team LLM synthesis call."""
    label = _label(team)
    header = f"Team {team} ({label}"
    if team_name:
        header += f" — {team_name}"
    header += ")"

    lines = [header, ""]
    n_total = sum(c["n_runs"] for c in clusters)
    lines.append(
        f"Ran {n_total} autonomous periods. Cluster analysis found "
        f"{len(clusters)} distinct routine{'s' if len(clusters) != 1 else ''}.\n"
    )

    # Sort clusters biggest-first
    for ci, cluster in enumerate(sorted(clusters, key=lambda c: c["n_runs"], reverse=True)):
        letter = chr(65 + ci)
        n = cluster["n_runs"]
        zone_str = (
            " → ".join(cluster["zone_sequence"]) if cluster["zone_sequence"] else "(no zones)"
        )
        delay_note = (
            f" (DELAYED start, ~{cluster['mean_delay_s']:.1f}s wait)" if cluster["delayed"] else ""
        )
        lines.append(f"Routine {letter} ({n}/{n_total} matches){delay_note}:")
        lines.append(f"  Matches: {', '.join(cluster['matches'])}")
        lines.append(f"  Zone sequence: {zone_str}")

        # Add per-match LLM descriptions for matches in this cluster
        cluster_matches = set(cluster["matches"])
        match_descs = [
            f"    [{m}] {info['name']}: {info['description']}"
            for m, st, info in match_interps
            if m in cluster_matches and info.get("description")
        ]
        if match_descs:
            lines.append("  Per-match descriptions:")
            lines.extend(match_descs[:6])  # cap to avoid prompt bloat
        lines.append("")

    lines.append(
        "Write a concise scouting entry for this team's autonomous. "
        "Cover each routine separately (label them A, B, C etc). "
        "State which is primary (most matches). "
        "Note delayed starts explicitly if present. "
        "Be concrete — focus on what zones the robot visits and what it likely does there. "
        "Max 250 words total."
    )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an FRC scouting analyst writing autonomous scouting entries.
You receive cluster analysis of a team's auto paths and per-match LLM descriptions.
Write a factual, concise scouting entry covering each distinct routine.
Format with markdown headers and bullet points. State match counts. No invented speculation.
If a delayed start is present, explain WHY it might be useful (alliance partner coordination).
Focus on what the robot does, not how it was tracked."""


def _generate_entry(team, team_name, clusters, match_interps, generate, model):
    user_msg = _build_synthesis_prompt(team, team_name, clusters, match_interps)
    for attempt_model in [model, FALLBACK_MODEL]:
        try:
            result = generate(
                model=attempt_model,
                system=SYSTEM_PROMPT,
                user=user_msg,
                timeout=TIMEOUT,
                think=False,
            )
            if result and len(result.strip()) > 50:
                return result.strip(), attempt_model
        except Exception:
            pass
    # Fallback: plain text from zone sequences
    lines = [f"**{_label(team)}** — {team_name}" if team_name else f"**{_label(team)}**", ""]
    for ci, c in enumerate(sorted(clusters, key=lambda x: x["n_runs"], reverse=True)):
        zs = " → ".join(c["zone_sequence"]) or "unknown"
        delay = f" (DELAYED ~{c['mean_delay_s']:.1f}s)" if c["delayed"] else ""
        lines.append(f"**Routine {chr(65 + ci)} ({c['n_runs']} runs){delay}:** {zs}")
    return "\n".join(lines), "heuristic"


def _write_team_md(team, team_name, clusters, match_interps, generate, model, force=False):
    os.makedirs(OUTDIR, exist_ok=True)
    team_num = team.replace("frc", "")
    out_path = os.path.join(OUTDIR, f"{team_num}.md")
    if not force and os.path.exists(out_path):
        return None, "skipped"

    entry, used_model = _generate_entry(team, team_name, clusters, match_interps, generate, model)

    # Build confidence breakdown
    source_counts = defaultdict(int)
    for _m, _st, info in match_interps:
        source_counts[info.get("source", "none")] += 1
    conf_str = "  ".join(f"{src}:{n}" for src, n in sorted(source_counts.items()))

    n_total = sum(c["n_runs"] for c in clusters)
    header = f"## {_label(team)}"
    if team_name:
        header += f" — {team_name}"

    md = "\n".join(
        [
            header,
            "",
            entry,
            "",
            f"*{n_total} matches | {len(clusters)} routine{'s' if len(clusters) != 1 else ''}"
            f" | Identity: {conf_str} | Model: {used_model}*",
            "",
            "---",
            "",
        ]
    )
    with open(out_path, "w") as f:
        f.write(md)
    return out_path, "ok"


def _try_pandoc(md_path, pdf_path):
    try:
        r = subprocess.run(
            [
                "pandoc",
                md_path,
                "-o",
                pdf_path,
                "--pdf-engine=xelatex",
                "-V",
                "geometry:margin=1in",
            ],
            capture_output=True,
            timeout=60,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--team", default="", help="single team number e.g. 8044")
    p.add_argument("--force", action="store_true", help="regenerate existing files")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--no-pdf", action="store_true", help="skip PDF conversion")
    a = p.parse_args()

    if not os.path.exists(ROUTES_JSON):
        raise SystemExit(f"{ROUTES_JSON} not found.\nRun: python team_routes.py --json-only")

    routes = json.load(open(ROUTES_JSON))
    interps = _load_per_match_interps()
    generate = _load_generate()

    os.makedirs(OUTDIR, exist_ok=True)

    if a.team:
        team_key = f"frc{a.team}" if not a.team.startswith("frc") else a.team
        teams = [team_key] if team_key in routes else []
        if not teams:
            raise SystemExit(f"Team {team_key} not found in team_routes.json")
    else:
        teams = sorted(
            routes.keys(),
            key=lambda t: int(t.replace("frc", "")) if t.replace("frc", "").isdigit() else 0,
        )

    ok = skip = fail = 0
    all_entries = []

    for team in teams:
        team_data = routes[team]
        clusters = team_data.get("clusters", [])
        if not clusters:
            continue
        team_name = _load_team_name(team)
        match_interps = interps.get(team, [])

        try:
            path, status = _write_team_md(
                team, team_name, clusters, match_interps, generate, a.model, a.force
            )
            if status == "ok":
                ok += 1
                all_entries.append(open(path).read())
                print(f"  {_label(team)}: written ({len(clusters)} routines)")
            elif status == "skipped":
                skip += 1
                team_num = team.replace("frc", "")
                cached = os.path.join(OUTDIR, f"{team_num}.md")
                if os.path.exists(cached):
                    all_entries.append(open(cached).read())
        except Exception as e:
            fail += 1
            import traceback

            print(f"  {_label(team)}: FAILED — {e}")
            traceback.print_exc()

    # Write combined _all.md
    all_path = os.path.join(OUTDIR, "_all.md")
    with open(all_path, "w") as f:
        f.write("# Auto Scouting Book\n\n")
        f.write(f"*{len(all_entries)} teams | Generated by auto_book.py*\n\n---\n\n")
        f.write("\n".join(all_entries))
    print(f"\nCombined report: {all_path}")

    # Try PDF
    if not a.no_pdf:
        pdf_path = os.path.join(OUTDIR, "_all.pdf")
        if _try_pandoc(all_path, pdf_path):
            print(f"PDF: {pdf_path}")
        else:
            print("pandoc not available or failed — markdown only (install pandoc for PDF)")

    print(f"\nDone: {ok} written, {skip} skipped, {fail} failed")
    if skip:
        print("  Use --force to regenerate all")
    print(f"\nOutput: {OUTDIR}/")


if __name__ == "__main__":
    main()
