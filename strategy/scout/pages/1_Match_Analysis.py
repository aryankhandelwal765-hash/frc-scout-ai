import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from scout.config import SYNTHESIS_MODEL
from scout.engine.prediction_engine import predict_match
from scout.engine.data_source import (
    is_reachable,
    is_configured,
    provider_name,
    get_event_matches,
)
from scout.engine.synthesis_engine import (
    build_strategy_brief,
    SYSTEM_PROMPT,
    stream_bot_analysis,
    stream_coaching,
)
from scout.engine.ollama_client import (
    is_running,
    list_models,
    stream_chat_verbose,
    stream_chat_with_history_verbose,
)
from scout.settings import get_my_team
from scout.stream_ui import render_stream_events

st.set_page_config(page_title="Match Analysis", page_icon="robot", layout="wide")

# ── Sidebar: Ollama model picker ───────────────────────────────────────────────

with st.sidebar:
    st.header("AI Settings")
    if not is_running():
        st.error("Ollama not running. Start it with: `ollama serve`")
        model = None
    else:
        models = list_models()
        if not models:
            st.warning("No models pulled yet.\nRun: `ollama pull qwen3.6:35b-a3b`")
            model = None
        else:
            # Default to the strategy-captain synthesis model when available.
            default_idx = models.index(SYNTHESIS_MODEL) if SYNTHESIS_MODEL in models else 0
            model = st.selectbox("Model", models, index=default_idx)

# ── Main ───────────────────────────────────────────────────────────────────────

st.title("Match Analysis")
st.caption("Enter 3 teams per alliance — or import a match from The Blue Alliance")

# Seed default team numbers once so the TBA importer can overwrite them via keys.
for _k, _dv in {"r1": 254, "r2": 1678, "r3": 2056, "b1": 1114, "b2": 148, "b3": 971}.items():
    st.session_state.setdefault(_k, _dv)

# ── Import a match from TBA ─────────────────────────────────────────────────────

with st.expander("📥 Import match from The Blue Alliance"):
    if provider_name() != "tba":
        st.caption("Match import requires the TBA provider (config.DATA_PROVIDER = 'tba').")
    else:
        ev_col, filt_col = st.columns([2, 1])
        with ev_col:
            event_key = st.text_input(
                "Event key", placeholder="e.g. 2026casj, 2026cmptx",
                help="Find it in the TBA event URL: thebluealliance.com/event/<key>",
            )
        with filt_col:
            mine_only = st.checkbox("Only my team's matches", value=bool(get_my_team()))

        if event_key:
            matches = get_event_matches(event_key.strip())
            if not matches:
                st.warning("No matches found (check the event key, or the schedule isn't posted yet).")
            else:
                my_team = get_my_team()
                if mine_only and my_team:
                    matches = [m for m in matches if my_team in m["red"] or my_team in m["blue"]]
                    if not matches:
                        st.info(f"No matches for team {my_team} at this event.")
                if matches:
                    def _fmt(m):
                        return f"{m['label']}: 🔴 {m['red']}  vs  🔵 {m['blue']}"
                    choice = st.selectbox("Match", matches, format_func=_fmt)
                    if st.button("Load teams into inputs"):
                        st.session_state["r1"], st.session_state["r2"], st.session_state["r3"] = choice["red"]
                        st.session_state["b1"], st.session_state["b2"], st.session_state["b3"] = choice["blue"]
                        st.rerun()

col_red, col_blue = st.columns(2, gap="large")

with col_red:
    st.markdown("### Red Alliance")
    r1 = st.number_input("Team 1", min_value=1, max_value=99999, key="r1")
    r2 = st.number_input("Team 2", min_value=1, max_value=99999, key="r2")
    r3 = st.number_input("Team 3", min_value=1, max_value=99999, key="r3")

with col_blue:
    st.markdown("### Blue Alliance")
    b1 = st.number_input("Team 1", min_value=1, max_value=99999, key="b1")
    b2 = st.number_input("Team 2", min_value=1, max_value=99999, key="b2")
    b3 = st.number_input("Team 3", min_value=1, max_value=99999, key="b3")

notes = st.text_area(
    "Scouting notes (optional)",
    placeholder="Pit/qualitative notes the numbers don't capture — e.g. '971 had a dead battery in quals, drivetrain fixed since.'",
    help="Free-text notes are fed to the strategy-captain layer alongside the quant data.",
)

include_bots = st.checkbox(
    "Include bot-by-bot analysis (6 quick per-robot reads)",
    help="Runs a short scouting read on each of the 6 robots using the faster bot model.",
)

predict_btn = st.button("Predict Match", type="primary", use_container_width=True)

if predict_btn:
    red_teams = [int(r1), int(r2), int(r3)]
    blue_teams = [int(b1), int(b2), int(b3)]

    with st.spinner("Fetching team data from Statbotics..."):
        result = predict_match(red_teams, blue_teams)
        brief = build_strategy_brief(result, notes=notes)

    missing = result["missing_teams"]
    all_teams = red_teams + blue_teams
    if missing:
        provider = provider_name().upper()
        if provider_name() == "tba" and not is_configured():
            st.error(
                "**No Blue Alliance API key configured** — add a free TBA read key as "
                "`TBA_AUTH_KEY` (in `scout/.streamlit/secrets.toml` or the env). "
                "See `secrets.toml.example`. The numbers below are zero placeholders; "
                "scouting notes still give a qualitative-only read."
            )
        elif len(missing) == len(all_teams) and not is_reachable():
            st.error(
                f"**{provider} data provider is unreachable** — every team returned no data. "
                "The numbers below are zero placeholders. "
                "Try again later, or add scouting notes for a qualitative-only read."
            )
        else:
            st.warning(f"No 2026 data found for teams: {missing} — using zero defaults.")

    # ── Win probability banner ─────────────────────────────────────────────────

    st.divider()

    red_pct = round(result["red_win_prob"] * 100, 1)
    blue_pct = round(result["blue_win_prob"] * 100, 1)
    diff = result["epa_diff"]

    banner_col1, banner_col2, banner_col3 = st.columns([2, 1, 2])
    with banner_col1:
        st.metric(
            label="Red Alliance Win Probability",
            value=f"{red_pct}%",
            delta=f"Predicted score: {result['red_predicted_score']:.0f} pts",
        )
    with banner_col2:
        color = "red" if diff > 0 else "dodgerblue"
        label = "RED" if diff > 0 else "BLUE"
        st.markdown(
            f"<h2 style='text-align:center;color:{color}'>{label} FAVORED</h2>"
            f"<p style='text-align:center'>by {abs(diff):.1f} EPA pts</p>",
            unsafe_allow_html=True,
        )
    with banner_col3:
        st.metric(
            label="Blue Alliance Win Probability",
            value=f"{blue_pct}%",
            delta=f"Predicted score: {result['blue_predicted_score']:.0f} pts",
        )

    # ── Team stats ─────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("Team Stats (2026 EPA)")

    stat_col_r, stat_col_b = st.columns(2)

    def render_team_table(stats_list):
        for s in stats_list:
            rank_str = f"#{s['rank']}" if s["rank"] else "N/A"
            pct_str = f"{s['percentile']*100:.1f}th pct" if s["percentile"] else ""
            st.markdown(
                f"**{s['team']} — {s['name']}** "
                f"<span style='color:gray;font-size:0.85em'>{rank_str} globally {pct_str}</span>",
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("EPA", f"{s['epa']:.1f}")
            c2.metric("Auto", f"{s['auto']:.1f}")
            c3.metric("Teleop", f"{s['teleop']:.1f}")
            c4.metric("Endgame", f"{s['endgame']:.1f}")
            wr = s["winrate"]
            w, l, t = s["wins"], s["losses"], s["ties"]
            st.caption(f"Win rate: {wr*100:.1f}%  ({w}-{l}-{t})")
            st.markdown("---")

    with stat_col_r:
        render_team_table(result["red_stats"])
    with stat_col_b:
        render_team_table(result["blue_stats"])

    # ── Quant layers: archetype / volatility / chemistry ───────────────────────

    st.divider()
    st.subheader("Strategic Profile")

    def render_quant(layers, chem_color):
        for a, v in zip(layers["archetypes"], layers["volatility"]["teams"]):
            st.markdown(
                f"**{a['team']} — {a['name']}** · "
                f"<span style='color:{chem_color}'>{a['tier']} {a['archetype']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Dominant phase: {a['dominant_phase']}  |  "
                f"{v['volatility_rating']} (consistency {v['consistency_score']:.1f}/10)  |  "
                f"floor ~{v['floor']:.0f} / ceiling ~{v['ceiling']:.0f}"
            )
        chem = layers["chemistry"]
        st.metric("Alliance Chemistry", f"{chem['overall_chemistry']:.1f}/10")
        vol = layers["volatility"]
        st.caption(
            f"Alliance floor ~{vol['alliance_floor']:.0f} / ceiling ~{vol['alliance_ceiling']:.0f} pts, "
            f"consistency {vol['alliance_consistency']:.1f}/10"
        )
        for note in chem["notes"]:
            st.caption(f"• {note}")

    quant_r, quant_b = st.columns(2)
    with quant_r:
        st.markdown("#### Red")
        render_quant(brief["red"], "red")
    with quant_b:
        st.markdown("#### Blue")
        render_quant(brief["blue"], "dodgerblue")

    # ── Strategy-captain read (Ollama synthesis) ───────────────────────────────

    st.divider()
    st.subheader("Strategy-Captain Read")

    if not model:
        st.info("Start Ollama and pick a model in the sidebar to enable the strategy read.")
    else:
        full_text = ""
        try:
            full_text = render_stream_events(
                stream_chat_verbose(model, SYSTEM_PROMPT, brief["text"])
            )
        except Exception as e:
            st.error(f"Ollama error: {e}")

        st.session_state["last_match_context"] = brief["text"]
        st.session_state["last_analysis"] = full_text

    # ── Coaching for MY team (from Settings) ────────────────────────────────────

    my_team = get_my_team()
    if model and my_team and my_team in (result["red_teams"] + result["blue_teams"]):
        side = "RED" if my_team in result["red_teams"] else "BLUE"
        st.divider()
        st.subheader(f"🎯 Coaching — how Team {my_team} wins (your {side} alliance)")
        st.caption("Tailored game plan for your team. Set/change your team on the Settings page.")
        try:
            render_stream_events(stream_coaching(result, brief, my_team, model=model))
        except Exception as e:
            st.error(f"Ollama error: {e}")

    # ── Bot-by-bot analysis (optional) ──────────────────────────────────────────

    if model and include_bots:
        st.divider()
        st.subheader("Bot-by-Bot Analysis")
        st.caption("A fast per-robot scouting read (uses the bot model).")
        all_stats = (
            [(s, "🔴") for s in result["red_stats"]]
            + [(s, "🔵") for s in result["blue_stats"]]
        )
        for stats, dot in all_stats:
            st.markdown(f"**{dot} {stats['team']} — {stats.get('name','Unknown')}**")
            try:
                render_stream_events(
                    stream_bot_analysis(stats),
                    target_tokens=90,
                    thinking_label="Scouting",
                )
            except Exception as e:
                st.error(f"Ollama error: {e}")

    # ── Chat ───────────────────────────────────────────────────────────────────

    if model and "last_match_context" in st.session_state:
        st.divider()
        st.subheader("Ask about this match")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_q := st.chat_input("Ask a follow-up question..."):
            st.session_state["chat_history"].append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            history = [
                {"role": "user", "content": (
                    f"Match context:\n{st.session_state['last_match_context']}\n\n"
                    f"Initial analysis:\n{st.session_state['last_analysis']}"
                )},
                {"role": "assistant", "content": st.session_state["last_analysis"]},
            ] + st.session_state["chat_history"]

            with st.chat_message("assistant"):
                reply = ""
                try:
                    reply = render_stream_events(
                        stream_chat_with_history_verbose(
                            model,
                            "You are an FRC strategy captain. Answer concisely using the provided match data.",
                            history,
                        ),
                        target_tokens=200,
                    )
                except Exception as e:
                    st.error(f"Ollama error: {e}")

            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
