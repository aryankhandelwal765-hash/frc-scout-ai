"""Synthesis engine — the strategy-captain reasoning layer.

This is the qualitative layer that yesterday's plan called for: take the quant
output (win-prob prediction + archetypes + chemistry + volatility) plus any
free-text scouting notes, and have qwen3.6:35b-a3b produce the kind of strategic
read a human strategy captain would give before a match.

Orchestrates:
    prediction_engine.predict_match   (structured: EPA win prob + scores)
    archetype_engine.classify_alliance (structured: playstyle per team)
    volatility_engine.profile_alliance (structured: consistency / floor-ceiling)
    chemistry_engine.analyze_chemistry (structured: alliance synergy)
    ollama_client                      (qwen3.6:35b-a3b synthesis)
"""

import json
import re

from scout.config import SYNTHESIS_MODEL, BOT_MODEL
from scout.engine.prediction_engine import predict_match
from scout.engine.archetype_engine import classify_alliance, classify_archetype
from scout.engine.volatility_engine import profile_alliance, profile_volatility
from scout.engine.chemistry_engine import analyze_chemistry
from scout.engine import ollama_client


SYSTEM_PROMPT = (
    "You are the strategy captain for a top FRC (FIRST Robotics Competition) team "
    "in the 2026 season. You read a match before it happens the way a veteran drive "
    "coach does: decisive, specific, and grounded in the data in front of you.\n\n"
    "You are given a full quantitative breakdown of a 3v3 match — EPA-based win "
    "probabilities and scores, each robot's playstyle archetype, each alliance's "
    "chemistry (role coverage, redundancy, top-heaviness), and each robot's "
    "volatility (consistency, floor, ceiling) — plus any human scouting notes.\n\n"
    "When a VISION-MEASURED SCORING block is present, treat those numbers as "
    "GROUND TRUTH from CV that watched actual matches — they are more accurate "
    "than aggregate OPR. Cite them explicitly (e.g. 'CV clocked 7287 at 120 balls/match') "
    "and prefer them over EPA when they disagree.\n\n"
    "Produce a strategy-captain read with these sections:\n"
    "1. **Call** — who is favored and how confident you are (reference the numbers).\n"
    "2. **Why** — the 2-3 dynamics that actually decide this match (archetype "
    "matchups, chemistry edges, phase battles like auto vs teleop). "
    "Endgame/climb is worth ~nothing this season — DO NOT mention it. "
    "Reference vision-measured scoring when it's present.\n"
    "3. **Path to victory** — one concrete plan for each alliance.\n"
    "4. **Swing factors & risk** — what would flip the result (volatility, "
    "top-heaviness, contested cycles, a boom-or-bust robot).\n\n"
    "Be concrete and reference specific teams and metrics. Do not restate the raw "
    "numbers as a list — interpret them. Keep it under 350 words."
)


def build_strategy_brief(prediction: dict, notes: str | None = None) -> dict:
    """Assemble all quant layers (+ optional notes) into a structured brief.

    Returns the structured pieces plus a formatted text block ready to hand to
    the LLM as the user message.
    """
    red_stats = prediction["red_stats"]
    blue_stats = prediction["blue_stats"]

    red_arch = classify_alliance(red_stats)
    blue_arch = classify_alliance(blue_stats)

    # Match-relative rank: 1 = strongest robot on the field, across all six.
    # This replaces the absolute Elite/Strong tier with a concrete, comparable
    # placement that a strategy captain actually reasons about.
    ranked = sorted(red_arch + blue_arch, key=lambda a: a.get("epa", 0.0), reverse=True)
    for i, a in enumerate(ranked, start=1):
        a["match_rank"] = i

    red_vol = profile_alliance(red_stats)
    blue_vol = profile_alliance(blue_stats)
    red_chem = analyze_chemistry(red_stats, red_arch)
    blue_chem = analyze_chemistry(blue_stats, blue_arch)

    red_pct = round(prediction["red_win_prob"] * 100, 1)
    blue_pct = round(prediction["blue_win_prob"] * 100, 1)

    def _alliance_block(name, teams, arch, vol, chem, score, pct):
        lines = [f"{name} ALLIANCE (teams {teams}) — win prob {pct}%, predicted {score:.0f} pts"]
        for a in arch:
            lines.append("  · " + a["summary"])
        for v in vol["teams"]:
            lines.append("  · " + v["summary"])
        lines.append(f"  · Chemistry: {chem['summary']}")
        lines.append(
            f"  · Alliance floor ~{vol['alliance_floor']:.0f} / "
            f"ceiling ~{vol['alliance_ceiling']:.0f} pts, "
            f"consistency {vol['alliance_consistency']:.1f}/10."
        )
        return "\n".join(lines)

    text = (
        "2026 FRC match analysis.\n\n"
        + _alliance_block(
            "RED",
            prediction["red_teams"],
            red_arch,
            red_vol,
            red_chem,
            prediction["red_predicted_score"],
            red_pct,
        )
        + "\n\n"
        + _alliance_block(
            "BLUE",
            prediction["blue_teams"],
            blue_arch,
            blue_vol,
            blue_chem,
            prediction["blue_predicted_score"],
            blue_pct,
        )
        + f"\n\nEPA edge: {'RED' if prediction['epa_diff'] > 0 else 'BLUE'} favored by "
        f"{abs(prediction['epa_diff']):.1f} EPA points."
    )

    # Vision-derived measurements (per-team ball scoring from CV). Runs after
    # aggregate stats so the LLM sees the aggregate + the measured behavior.
    # Only inserted for teams that have any vision data — degrades silently.
    from scout.engine.vision_loader import get_team_vision, get_team_auto_routes

    vision_lines = []
    auto_lines = []
    for team in prediction["red_teams"] + prediction["blue_teams"]:
        v = get_team_vision(team)
        if v is None or v.get("matches_count", 0) == 0:
            continue
        confidence = ""
        if v["id_confidence"] < 0.7:
            confidence = " (id uncertain)"
        vision_lines.append(
            f"  · Team {team}: {v['avg_balls_scored']} avg balls scored/match "
            f"(peak {v['max_balls_scored']}) across {v['matches_count']} vision-scouted matches{confidence}"
        )
        # Auto route context from interpret_auto.py
        routes = get_team_auto_routes(team)
        if routes:
            total = sum(r["count"] for r in routes)
            parts = []
            for r in routes[:3]:
                delayed = " (delayed)" if r["delayed"] else ""
                zs = " → ".join(r["zone_seq"][:4]) if r["zone_seq"] else ""
                zone_note = f" [{zs}]" if zs else ""
                parts.append(f"{r['name']}{delayed} ×{r['count']}/{total}{zone_note}")
            auto_lines.append(f"  · Team {team} AUTO: {' | '.join(parts)}")
    if vision_lines:
        text += "\n\nVISION-MEASURED SCORING (from CV, per robot):\n" + "\n".join(vision_lines)
    if auto_lines:
        text += "\n\nAUTO ROUTES (CV-tracked zone sequences):\n" + "\n".join(auto_lines)

    if prediction.get("missing_teams"):
        text += (
            f"\n\nNOTE: no 2026 data for teams {prediction['missing_teams']} — "
            "treated as zero-EPA; weight conclusions accordingly."
        )

    if notes and notes.strip():
        text += (
            f"\n\nHUMAN SCOUTING NOTES (qualitative, may override the numbers):\n{notes.strip()}"
        )

    return {
        "text": text,
        "red": {"archetypes": red_arch, "volatility": red_vol, "chemistry": red_chem},
        "blue": {"archetypes": blue_arch, "volatility": blue_vol, "chemistry": blue_chem},
    }


def synthesize(prediction: dict, notes: str | None = None, model: str | None = None) -> dict:
    """Full non-streaming strategy read. Returns the brief + the LLM text."""
    brief = build_strategy_brief(prediction, notes)
    model = model or SYNTHESIS_MODEL
    read = ollama_client.generate(model, SYSTEM_PROMPT, brief["text"])
    return {"model": model, "brief": brief, "read": read}


def stream_synthesis(prediction: dict, notes: str | None = None, model: str | None = None):
    """Stream the strategy read token-by-token (drives the NDJSON API stream).

    Yields text chunks; build the brief first via build_strategy_brief if you
    also need the structured layers.
    """
    brief = build_strategy_brief(prediction, notes)
    model = model or SYNTHESIS_MODEL
    yield from ollama_client.stream_chat(model, SYSTEM_PROMPT, brief["text"])


def analyze_matchup(
    red_teams: list[int],
    blue_teams: list[int],
    year: int = 2026,
    notes: str | None = None,
    model: str | None = None,
) -> dict:
    """End-to-end: predict + quant engines + strategy-captain synthesis.

    The single entry point a thin API or UI can call for a full read.
    """
    prediction = predict_match(red_teams, blue_teams, year=year)
    result = synthesize(prediction, notes=notes, model=model)
    return {"prediction": prediction, **result}


# ── Bot-by-bot analysis ─────────────────────────────────────────────────────────

BOT_SYSTEM_PROMPT = (
    "You are an FRC scout giving a fast tactical read on ONE robot for a drive team. "
    "You are given the robot's real numbers. Reply in EXACTLY 3 lines, one short sentence "
    "each (about 10-14 words), no preamble, no headings, no bullet symbols:\n"
    "Line 1 — its role, as a short phrase built from its Playstyle and Main phase.\n"
    "Line 2 — its scoring level: cite its avg points, its floor and ceiling, and whether it "
    "is consistent or high-variance (match the Reliability note given).\n"
    "Line 3 — the single best way the opposing alliance should limit it.\n"
    "Rules: use ONLY the numbers given; 'Avg points' is the average and the floor is a "
    "bad-day low (NOT the average); never call a share a 'score'; never call a high-variance "
    "robot reliable; NEVER mention endgame or climbing — worth ~zero this season."
)


def build_bot_brief(
    stats: dict, archetype: dict | None = None, volatility: dict | None = None
) -> str:
    """One robot's data as a compact, correctly-labeled prompt for per-bot analysis."""
    a = archetype or classify_archetype(stats)
    v = volatility or profile_volatility(stats)
    shares = a["phase_shares"]
    rating = v["volatility_rating"]
    if rating == "Boom-or-Bust":
        reliability = "HIGH-VARIANCE and swingy — cannot be counted on; some matches near the floor"
    elif rating == "Streaky":
        reliability = "somewhat inconsistent"
    else:
        reliability = "consistent"
    return (
        f"Team {stats['team']} ({stats.get('name', 'Unknown')})\n"
        f"Playstyle: {a['archetype']} — Main phase: {a['dominant_phase']}\n"
        f"Scoring split: auto {shares['auto'] * 100:.0f}% / "
        f"teleop {shares['teleop'] * 100:.0f}%\n"
        f"Avg points added: {stats['epa']:.0f} per match "
        f"(bad-day floor ~{v['floor']:.0f}, best-day ceiling ~{v['ceiling']:.0f})\n"
        f"Reliability: {reliability} (consistency {v['consistency_score']:.1f}/10)\n"
        f"Record: {stats['wins']}-{stats['losses']}-{stats['ties']} "
        f"({stats['winrate'] * 100:.0f}% wins)"
    )


def stream_bot_analysis(
    stats: dict,
    archetype: dict | None = None,
    volatility: dict | None = None,
    model: str | None = None,
    think: bool = False,
):
    """Stream a short per-robot read (3 tight lines). Uses BOT_MODEL, disables
    thinking, and hard-caps length via num_predict so it stays brief."""
    brief = build_bot_brief(stats, archetype, volatility)
    model = model or BOT_MODEL
    yield from ollama_client.stream_chat_verbose(
        model, BOT_SYSTEM_PROMPT, brief, think=think, options={"num_predict": 150}
    )


# ── Per-robot main job (fast, structured) ────────────────────────────────────────

ROLES_SYSTEM_PROMPT = (
    "You are an FRC strategy captain assigning every robot in an upcoming 2026 "
    "3v3 match its single MAIN JOB. You are given all six robots ranked #1 "
    "(strongest) to #6 by OPR contribution, with each robot's auto/teleop scoring "
    "split and which alliance it is on. Endgame/climb is worth ~zero this season "
    "— do NOT assign climb roles.\n\n"
    "For EACH robot, give ONE short imperative job phrase (max 8 words) describing "
    "what that robot should focus on this match. Examples: 'Primary high-goal cycler', "
    "'Fast auto, then feed the #1 robot', 'Defend their #1 scorer', "
    "'Secondary cycler, protect the ball path', 'Ball-thief on the top opponent'.\n\n"
    "Return ONLY a JSON object mapping each team number (as a string key) to its "
    "job phrase (string). No commentary, no markdown, no extra keys."
)


def build_roles_brief(prediction: dict, brief: dict) -> str:
    """Compact all six robots (with match rank + phase split) into one prompt."""

    def block(side: str, arch: list[dict]) -> list[str]:
        out = []
        for a in arch:
            s = a["phase_shares"]
            out.append(
                f"  [{side.upper()}] #{a.get('match_rank', '?')} "
                f"Team {a['team']} ({a.get('name', 'Unknown')}): "
                f"OPR {a['epa']:.0f}, "
                f"auto {s['auto'] * 100:.0f}% / teleop {s['teleop'] * 100:.0f}%"
            )
        return out

    lines = block("red", brief["red"]["archetypes"]) + block("blue", brief["blue"]["archetypes"])
    return (
        "Six robots in a 2026 FRC match, ranked #1 (strongest) to #6 by OPR:\n"
        + "\n".join(lines)
        + "\n\nAssign each robot its single main job as a JSON {team: job} object."
    )


def _parse_roles_json(raw: str) -> dict:
    """Best-effort parse of the model's JSON object (tolerates stray text)."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def generate_roles(prediction: dict, brief: dict | None = None, model: str | None = None) -> dict:
    """Return {team_number(str): main-job phrase(str)} for all six robots.

    One batched call to the fast bot model (thinking disabled), JSON-formatted,
    so the whole field gets a job in a couple seconds.
    """
    brief = brief or build_strategy_brief(prediction)
    model = model or BOT_MODEL
    user = build_roles_brief(prediction, brief)
    raw = ollama_client.generate(model, ROLES_SYSTEM_PROMPT, user, fmt="json", think=False)
    parsed = _parse_roles_json(raw)
    # Normalize: keys -> str team numbers, values -> trimmed str.
    return {str(k): str(v).strip() for k, v in parsed.items() if str(v).strip()}


# ── My-team coaching ────────────────────────────────────────────────────────────


def coach_system_prompt(my_team: int, alliance: str) -> str:
    return (
        f"You are the strategy captain and drive coach for FRC Team {my_team}, on the "
        f"{alliance} alliance in this match. Your job: tell Team {my_team} EXACTLY what "
        f"to do to maximize its chance of winning THIS match. Give a directive game plan:\n"
        f"1. **Your role** — what {my_team} should focus on given its own strengths and "
        f"its two alliance partners (avoid overlap, cover gaps).\n"
        f"2. **Phase plan** — concrete priorities for auto and teleop. "
        f"(Endgame/climb is worth ~zero this season — skip it.)\n"
        f"3. **Exploit / avoid** — which specific opponents to pressure or defend, and "
        f"which to stay away from.\n"
        f"4. **Biggest risk** — the one thing most likely to cost {my_team} the match.\n"
        f"Be specific, reference team numbers and the metrics. Speak directly to "
        f"Team {my_team} ('you'). Under 300 words."
    )


def build_coach_brief(prediction: dict, brief: dict, my_team: int) -> tuple[str, str]:
    """Return (system_prompt, user_text) for a coaching read for my_team.

    Raises ValueError if my_team is not in the match.
    """
    if my_team in prediction["red_teams"]:
        alliance = "RED"
    elif my_team in prediction["blue_teams"]:
        alliance = "BLUE"
    else:
        raise ValueError(f"Team {my_team} is not in this match.")
    system = coach_system_prompt(my_team, alliance)
    user = (
        brief["text"] + f"\n\nYou are coaching Team {my_team} (on the {alliance} alliance). "
        f"Give {my_team}'s winning game plan."
    )
    return system, user


def stream_coaching(prediction: dict, brief: dict, my_team: int, model: str | None = None):
    """Stream a coaching read tailored to my_team. Uses the 35b model by default."""
    system, user = build_coach_brief(prediction, brief, my_team)
    model = model or SYNTHESIS_MODEL
    yield from ollama_client.stream_chat_verbose(model, system, user)


# ── Captain's briefing (alliance-level) ─────────────────────────────────────────


def captain_system_prompt(
    our_teams: list[int], opp_teams: list[int], our_side: str, opp_side: str
) -> str:
    return (
        f"You are captain of the {our_side.upper()} alliance ({our_teams}), vs "
        f"{opp_side.upper()} ({opp_teams}). Deliver a TIGHT pre-match briefing to "
        f"your drivers. Speak as '{our_side.upper()}' and 'we/our'. NEVER refer to "
        f"the opposing side as if it were yours.\n\n"
        f"Format — EXACTLY these three headers, one short paragraph each, no bullet "
        f"lists, no filler, no restating input numbers as lists:\n\n"
        f"1. **Game Plan** — 2-3 sentences: our overall strategy + who owns auto vs teleop.\n"
        f"2. **Per-Robot Jobs** — one line per robot: '<team>: <one action>'.\n"
        f"3. **Their Weakness** — 1-2 sentences: their single biggest weakness + how we exploit it.\n\n"
        f"Endgame/climb is worth ~nothing this season — ignore it unless it's decisive. "
        f"Cite vision-measured ball counts when available. HARD CAP: 180 words."
    )


def build_captain_brief(prediction: dict, brief: dict, our_side: str) -> tuple[str, str]:
    """Return (system_prompt, user_text) for a captain's briefing.

    our_side is 'red' or 'blue' — the alliance the briefing is written for.
    """
    if our_side == "red":
        our, opp, our_lbl, opp_lbl = (
            prediction["red_teams"],
            prediction["blue_teams"],
            "RED",
            "BLUE",
        )
    else:
        our, opp, our_lbl, opp_lbl = (
            prediction["blue_teams"],
            prediction["red_teams"],
            "BLUE",
            "RED",
        )
    system = captain_system_prompt(our, opp, our_lbl, opp_lbl)
    user = (
        brief["text"] + f"\n\nYou are captaining the {our_lbl} alliance ({our}). "
        f"Give the briefing for {our_lbl}."
    )
    return system, user


def stream_captain(prediction: dict, brief: dict, our_side: str, model: str | None = None):
    """Stream an alliance captain's briefing. Uses the 35b model by default."""
    system, user = build_captain_brief(prediction, brief, our_side)
    model = model or SYNTHESIS_MODEL
    yield from ollama_client.stream_chat_verbose(model, system, user)
