"""
Turn the measured numbers into readable scouting notes, using a local Ollama model.

The model is a WRITER, not an analyst. Every number handed to it is already measured
and already carries a provenance tag, and the prompt forbids inventing anything not
in the fact block. That matters more than usual here: a fluent model asked to
"describe a team" will happily produce plausible sentences about driver skill or
defense that nothing in this pipeline measured.

Provenance is passed through to the prose so a reader can tell what is official from
what is inferred:
  tba     official Blue Alliance data
  split   a TBA alliance total apportioned by vision -- total exact, split estimated
  vision  measured from broadcast video only, carries identity/detection error
Anything the pipeline cannot measure is simply absent, and the model is told to say
so rather than fill the gap.

PyCharm: set TEAM (or "" for all), click Run.
"""

import argparse
import collections
import csv
import glob
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUTDIR = os.path.join(HERE, "reports")

# ===========================================================================
# EDIT ME (PyCharm one-click).
# ===========================================================================
TEAM = ""  # e.g. "2056"; "" = every team
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
TABLE = os.path.join(HERE, "scout_table.csv")
TEMPERATURE = 0.2  # low: this is reporting, not creative writing
TIMEOUT = 180
DRY_RUN = False  # True = print the fact block, never call the model
# ===========================================================================

SYSTEM = """You write short scouting notes for an FRC team from MEASURED data.

Hard rules:
- Use ONLY the facts given. Never invent a number, a behaviour, or an opinion.
- Every fact carries a source tag. Reflect its reliability in how you word it:
    [tba]    official data. State it plainly as fact.
    [split]  an alliance total divided up by video analysis. The total is exact,
             the per-robot split is an ESTIMATE. Hedge accordingly.
    [vision] measured from broadcast video only. Say "video shows" or similar, and
             mention when identification confidence is low.
- If something is not in the facts (driver skill, defense, reliability, intent),
  DO NOT comment on it. Never speculate about why a robot did something.
- No praise, no hype, no filler. A scout reads this to make a pick decision.
- Do NOT reinterpret a fact into something it does not say. Counts of alliance
  colour or driver station are NOT win-loss records or scores. If a fact states
  what it is not, respect that.
- 3 to 5 sentences. Plain prose, no headings, no bullet points, no markdown.
"""


def load_table(path=TABLE):
    if not os.path.exists(path):
        raise SystemExit(f"No {path} -- run scout_table.py first.")
    with open(path) as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def team_facts(team, rows, routes=None):
    """A compact, fully-sourced fact block for one team."""
    mine = [r for r in rows if r["team"] == str(team)]
    if not mine:
        return None
    n = len(mine)
    climbs = sum(int(r["climb"] or 0) for r in mine)
    autoclimbs = sum(int(r["auto_climb"] or 0) for r in mine)
    idsrc = collections.Counter(r["identity_src"] for r in mine if r["identity_src"])
    allies = collections.Counter(r["alliance"] for r in mine)
    stations = collections.Counter(r["station"] for r in mine)

    def avg(col):
        v = [_num(r[col]) for r in mine]
        v = [x for x in v if x is not None]
        return round(sum(v) / len(v), 1) if v else None

    # Phrasing is deliberately verbose and self-describing. A terse label like
    # "alliance: blue 10, red 8" reads as a WIN-LOSS RECORD to a language model --
    # it did exactly that on the first run. Every count here says what it counts.
    st = ", ".join(
        f"station {k} in {v} match{'es' if v != 1 else ''}" for k, v in sorted(stations.items())
    )
    f = [
        f"[tba] played {n} matches in total",
        f"[tba] was on the BLUE alliance for {allies['blue']} of those matches and "
        f"the RED alliance for {allies['red']} (this is which side of the field, "
        f"NOT a win-loss record -- this pipeline does not measure wins)",
        f"[tba] started from {st} (driver station position, not a score)",
        f"[tba] climbed the tower in endgame in {climbs} of {n} matches",
        f"[tba] climbed during autonomous in {autoclimbs} of {n} matches",
    ]
    for col, label in (
        ("alliance_auto_fuel", "auto"),
        ("alliance_teleop_fuel", "teleop"),
        ("alliance_total_fuel", "total"),
    ):
        a = avg(col)
        if a is not None:
            f.append(
                f"[tba] mean ALLIANCE {label} fuel in this team's matches: {a} "
                f"(this is the 3-robot alliance total, NOT this team alone)"
            )
    if idsrc:
        conf = idsrc.get("read", 0) + idsrc.get("elim", 0)
        f.append(
            f"[vision] team identification: {idsrc.get('read', 0)} by reading the "
            f"bumper number, {idsrc.get('elim', 0)} by elimination, "
            f"{idsrc.get('rank', 0)} by unverified station-order guess"
        )
        if conf < n / 2:
            f.append(
                "[vision] WARNING: most matches were identified by the unverified "
                "station-order guess, so video-derived facts may describe the "
                "wrong robot"
            )
    if routes:
        f.append(
            f"[vision] auto routes: {routes['n_runs']} usable autos forming "
            f"{routes['n_groups']} distinct starting group(s)"
        )
    return {"team": str(team), "matches": n, "facts": f}


def load_routes():
    """n usable autos and start-group count per team, from team_routes if built."""
    try:
        sys.path.insert(0, HERE)
        import topdown as td
        import team_routes as tr

        mapper = td.load_mapper()
        if mapper is None:
            return {}
        per = tr.load_runs(mapper)
        out = {}
        for team, runs in per.items():
            if len(runs) < 2:
                continue
            out[team.replace("frc", "")] = {
                "n_runs": len(runs),
                "n_groups": len(tr.cluster_starts(runs)),
            }
        return out
    except Exception as e:
        print(f"  (route data unavailable: {e.__class__.__name__}: {e})")
        return {}


def ask(facts, host=OLLAMA_HOST, model=MODEL, temperature=TEMPERATURE, timeout=TIMEOUT):
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"Team {facts['team']}. Facts:\n" + "\n".join(facts["facts"]),
                },
            ],
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--team", default=TEAM)
    p.add_argument("--host", default=OLLAMA_HOST)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    a = p.parse_args()

    rows = load_table()
    routes = load_routes()
    teams = [a.team] if a.team else sorted({r["team"] for r in rows}, key=lambda t: int(t))
    print(
        f"{len(teams)} teams, model={a.model} host={a.host}" + ("  [DRY RUN]" if a.dry_run else "")
    )

    os.makedirs(OUTDIR, exist_ok=True)
    combined = []
    for i, t in enumerate(teams, 1):
        f = team_facts(t, rows, routes.get(t))
        if not f:
            continue
        if a.dry_run:
            print(f"\n--- {t}\n" + "\n".join(f["facts"]))
            continue
        try:
            text = ask(f, a.host, a.model)
        except urllib.error.URLError as e:
            raise SystemExit(
                f"Cannot reach Ollama at {a.host}: {e.reason}\n"
                "Start it with:  OLLAMA_HOST=0.0.0.0 ollama serve"
            )
        except Exception as e:
            print(f"  {t}: {e.__class__.__name__}: {e}")
            continue
        open(os.path.join(OUTDIR, f"{t}.md"), "w").write(
            f"# Team {t}\n\n{text}\n\n## Measured facts\n\n"
            + "\n".join(f"- {x}" for x in f["facts"])
            + "\n"
        )
        combined.append(f"## Team {t}\n\n{text}\n")
        print(f"  [{i}/{len(teams)}] {t}")
    if combined:
        open(os.path.join(OUTDIR, "_all.md"), "w").write(
            "# 2026hop scouting notes\n\n"
            "Generated from measured data. Every statement traces to a fact block in "
            "the per-team file; nothing here is human judgement.\n\n" + "\n".join(combined)
        )
        print(f"\nwrote {len(combined)} reports -> {OUTDIR}")


if __name__ == "__main__":
    main()
