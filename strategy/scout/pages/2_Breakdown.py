import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from scout.config import SYNTHESIS_MODEL
from scout.engine.data_source import get_team_stats, is_reachable, is_configured, provider_name
from scout.engine.ollama_client import (
    is_running,
    list_models,
    stream_chat_verbose,
    stream_chat_with_history_verbose,
)
from scout.stream_ui import render_stream_events

st.set_page_config(page_title="Team Breakdown", page_icon="robot", layout="wide")

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
            default_idx = models.index(SYNTHESIS_MODEL) if SYNTHESIS_MODEL in models else 0
            model = st.selectbox("Model", models, index=default_idx)

# ── Main ───────────────────────────────────────────────────────────────────────

st.title("Team Breakdown")
st.caption("2026 season stats + AI scouting report — data from Statbotics")

team_num = st.number_input("Team Number", min_value=1, max_value=99999, value=254)
search_btn = st.button("Look up Team", type="primary")

if search_btn:
    with st.spinner(f"Fetching 2026 data for team {team_num}..."):
        stats = get_team_stats(int(team_num))

    if stats is None:
        provider = provider_name().upper()
        if provider_name() == "tba" and not is_configured():
            st.error(
                "**No Blue Alliance API key configured** — add a free TBA read key as "
                "`TBA_AUTH_KEY` (in `scout/.streamlit/secrets.toml` or the env). "
                "See `secrets.toml.example`."
            )
        elif not is_reachable():
            st.error(f"**{provider} data provider is unreachable** — try again later.")
        else:
            st.error(f"Team {team_num} not found in {provider} 2026 data.")
        st.stop()

    rank_str = f"#{stats['rank']} globally" if stats["rank"] else ""
    pct_str = (
        f"({stats['percentile']*100:.1f}th percentile out of {stats['team_count']} teams)"
        if stats["percentile"] else ""
    )

    st.markdown(f"## Team {stats['team']} — {stats['name']}")
    st.caption(f"{rank_str} {pct_str}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("EPA", f"{stats['epa']:.1f}", help="Mean total EPA")
    m2.metric("Auto", f"{stats['auto']:.1f}")
    m3.metric("Teleop", f"{stats['teleop']:.1f}")
    m4.metric("Endgame", f"{stats['endgame']:.1f}")
    m5.metric("Win Rate", f"{stats['winrate']*100:.1f}%",
              help=f"{stats['wins']}-{stats['losses']}-{stats['ties']}")

    st.caption(f"EPA std dev: ±{stats['epa_sd']:.1f}  |  Season peak: {stats['epa_max']:.1f}")

    # ── Phase breakdown ────────────────────────────────────────────────────────

    st.divider()
    st.subheader("Score Phase Breakdown")

    total = stats["auto"] + stats["teleop"] + stats["endgame"]
    auto_pct = teleop_pct = endgame_pct = 0.0
    if total > 0:
        auto_pct = stats["auto"] / total * 100
        teleop_pct = stats["teleop"] / total * 100
        endgame_pct = stats["endgame"] / total * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Auto share", f"{auto_pct:.0f}%")
        c2.metric("Teleop share", f"{teleop_pct:.0f}%")
        c3.metric("Endgame share", f"{endgame_pct:.0f}%")

        chart_data = pd.DataFrame({
            "Phase": ["Auto", "Teleop", "Endgame"],
            "EPA": [stats["auto"], stats["teleop"], stats["endgame"]],
        })
        st.bar_chart(chart_data.set_index("Phase"))

    # ── Record ─────────────────────────────────────────────────────────────────

    st.divider()
    w, l, t = stats["wins"], stats["losses"], stats["ties"]
    total_matches = w + l + t
    st.subheader("2026 Record")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Wins", w)
    rc2.metric("Losses", l)
    rc3.metric("Ties", t)
    rc4.metric("Matches Played", total_matches)

    # ── AI Scouting Report ─────────────────────────────────────────────────────

    st.divider()
    st.subheader("AI Scouting Report")

    team_context = (
        f"Team {stats['team']} — {stats['name']} (2026 FRC Season)\n"
        f"Global rank: {rank_str} {pct_str}\n\n"
        f"EPA breakdown:\n"
        f"  Total EPA: {stats['epa']:.1f} (±{stats['epa_sd']:.1f} std dev, peak: {stats['epa_max']:.1f})\n"
        f"  Auto:      {stats['auto']:.1f}  ({auto_pct:.0f}% of scoring)\n"
        f"  Teleop:    {stats['teleop']:.1f}  ({teleop_pct:.0f}% of scoring)\n"
        f"  Endgame:   {stats['endgame']:.1f}  ({endgame_pct:.0f}% of scoring)\n\n"
        f"Record: {w}-{l}-{t} ({stats['winrate']*100:.1f}% win rate) over {total_matches} matches."
    )

    if not model:
        st.info("Start Ollama and pick a model in the sidebar to enable AI scouting reports.")
    else:
        system_prompt = (
            "You are an expert FRC (FIRST Robotics Competition) scout. "
            "Given a team's 2026 EPA statistics, write a concise scouting report covering: "
            "1) Team strengths based on their phase breakdown, "
            "2) Potential weaknesses or inconsistencies, "
            "3) How reliable they are as an alliance partner, "
            "4) What kind of alliance they pair well with. "
            "Be specific to the numbers. Keep it under 180 words."
        )

        full_report = ""
        try:
            full_report = render_stream_events(
                stream_chat_verbose(model, system_prompt, team_context),
                target_tokens=250,
            )
        except Exception as e:
            st.error(f"Ollama error: {e}")

        # ── Follow-up chat ─────────────────────────────────────────────────────

        st.divider()
        st.subheader(f"Ask about Team {stats['team']}")

        chat_key = f"team_chat_{stats['team']}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_q := st.chat_input(f"Ask about team {stats['team']}..."):
            st.session_state[chat_key].append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            history = [
                {"role": "user", "content": team_context},
                {"role": "assistant", "content": full_report},
            ] + st.session_state[chat_key]

            with st.chat_message("assistant"):
                reply = ""
                try:
                    reply = render_stream_events(
                        stream_chat_with_history_verbose(
                            model,
                            "You are an expert FRC scout. Answer concisely using the team's 2026 data.",
                            history,
                        ),
                        target_tokens=200,
                    )
                except Exception as e:
                    st.error(f"Ollama error: {e}")

            st.session_state[chat_key].append({"role": "assistant", "content": reply})
