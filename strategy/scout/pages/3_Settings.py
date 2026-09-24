import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from scout.settings import get_my_team, set_my_team
from scout.engine import data_source, ollama_client
from scout.config import SYNTHESIS_MODEL, BOT_MODEL

st.set_page_config(page_title="Settings", page_icon="robot", layout="wide")

st.title("Settings")

# ── My team ─────────────────────────────────────────────────────────────────────

st.subheader("My Team")
st.caption(
    "Set your team number. Any match where your team appears will get a dedicated "
    "coaching read — the best moves *your team* should make to win."
)

current = get_my_team()
col1, col2 = st.columns([3, 1])
with col1:
    team_val = st.number_input(
        "Team number",
        min_value=0, max_value=99999,
        value=current if current is not None else 0,
        help="Set to 0 to clear.",
    )
with col2:
    st.write("")
    st.write("")
    if st.button("Save", type="primary", use_container_width=True):
        set_my_team(int(team_val) if team_val > 0 else None)
        st.success(f"Saved: team {int(team_val)}" if team_val > 0 else "Cleared my team.")
        st.rerun()

if current is not None:
    st.info(f"Current: **Team {current}** — coaching reads are enabled for its matches.")
else:
    st.warning("No team set — set one to enable coaching reads.")

# ── Status ──────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("System Status")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Data provider**")
    st.write(f"Provider: `{data_source.provider_name()}`")
    st.write(f"Configured (API key): {'✅' if data_source.is_configured() else '❌'}")
    st.write(f"Reachable: {'✅' if data_source.is_reachable() else '❌'}")
    if data_source.provider_name() == "tba" and not data_source.is_configured():
        st.caption("Add TBA_AUTH_KEY to secrets.toml or env (see secrets.toml.example).")
with c2:
    st.markdown("**Ollama**")
    running = ollama_client.is_running()
    st.write(f"Running: {'✅' if running else '❌'}")
    st.write(f"Strategy model: `{SYNTHESIS_MODEL}`")
    st.write(f"Bot-analysis model: `{BOT_MODEL}`")
    if running:
        st.caption("Available: " + ", ".join(ollama_client.list_models()))
